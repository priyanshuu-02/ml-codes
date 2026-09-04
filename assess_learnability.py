"""
What, precisely, can be learned from this IMU?

Training produced a clear negative result on the headline targets: the model ties or loses
to persistence on speed, forward displacement, lateral displacement and heading, both when
holding out whole sessions AND when splitting a single session temporally. Within-session
failure rules out phone-mount variation as the explanation, so the honest question is which
targets carry learnable signal at all.

This evaluates the trained checkpoints against two baselines per target:

  constant    : always emit the training mean. Ignores the inputs entirely.
  persistence : assume nothing changed during the window. Speed stays at the last known
                value, forward displacement is that speed times the span, heading does not
                turn, lateral stays zero. This is what a constant-velocity INS already
                does without any network, so a model must beat it to be worth deploying.

Reporting a model against the weaker baseline only would be flattering and misleading.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import WINDOW_SPAN_SECONDS, build_split
from src.models.idr_v1 import IdrV1Model
from train_idr import temporal_resplit


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IdrV1Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, split, normalizer, device, batch_size=4096):
    imu = normalizer.imu(split["imu"]).astype(np.float32)
    initial = normalizer.speed_forward(split["initial_speed"]).astype(np.float32)

    speeds, accelerations, positions, headings, motions = [], [], [], [], []
    for start in range(0, len(imu), batch_size):
        chunk = torch.from_numpy(imu[start:start + batch_size]).to(device)
        chunk_speed = torch.from_numpy(initial[start:start + batch_size]).to(device)
        outputs = model(chunk, chunk_speed)
        speeds.append(normalizer.speed_inverse(outputs[0].cpu().numpy()))
        accelerations.append(normalizer.acceleration_inverse(outputs[2].cpu().numpy()))
        positions.append(normalizer.position_inverse(outputs[4].cpu().numpy()))
        headings.append(outputs[6].cpu().numpy())
        motions.append(outputs[8].argmax(dim=1).cpu().numpy())

    return {
        "speed": np.concatenate(speeds),
        "acceleration": np.concatenate(accelerations),
        "position": np.concatenate(positions),
        "heading_delta": np.concatenate(headings),
        "motion": np.concatenate(motions),
    }


def mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def report(name, train_split, split, prediction):
    truth_speed = split["speed"]
    truth_acceleration = split["acceleration"]
    truth_forward = split["position"][:, 0]
    truth_lateral = split["position"][:, 1]
    truth_heading = split["heading_delta"]
    initial_speed = split["initial_speed"]

    rows = [
        ("speed (m/s)",
         mae(prediction["speed"], truth_speed),
         mae(np.full_like(truth_speed, train_split["speed"].mean()), truth_speed),
         mae(initial_speed, truth_speed)),
        ("acceleration (m/s2)",
         mae(prediction["acceleration"], truth_acceleration),
         mae(np.full_like(truth_acceleration, train_split["acceleration"].mean()), truth_acceleration),
         mae(np.zeros_like(truth_acceleration), truth_acceleration)),
        ("forward (m)",
         mae(prediction["position"][:, 0], truth_forward),
         mae(np.full_like(truth_forward, train_split["position"][:, 0].mean()), truth_forward),
         mae(initial_speed * WINDOW_SPAN_SECONDS, truth_forward)),
        ("lateral (m)",
         mae(prediction["position"][:, 1], truth_lateral),
         mae(np.full_like(truth_lateral, train_split["position"][:, 1].mean()), truth_lateral),
         mae(np.zeros_like(truth_lateral), truth_lateral)),
        ("heading delta (rad)",
         mae(prediction["heading_delta"], truth_heading),
         mae(np.full_like(truth_heading, train_split["heading_delta"].mean()), truth_heading),
         mae(np.zeros_like(truth_heading), truth_heading)),
    ]

    majority = np.bincount(train_split["motion"], minlength=3).argmax()
    motion_accuracy = float((prediction["motion"] == split["motion"]).mean())
    majority_accuracy = float((split["motion"] == majority).mean())

    print(f"\n--- {name}  ({len(split['imu']):,} windows) ---")
    print(f"{'target':<22} {'model':>9} {'constant':>10} {'persist':>9} "
          f"{'vs const':>9} {'vs persist':>11}  verdict")
    results = {}
    for label, model_value, constant_value, persistence_value in rows:
        gain_constant = (constant_value - model_value) / max(constant_value, 1e-9) * 100
        gain_persistence = (persistence_value - model_value) / max(persistence_value, 1e-9) * 100
        verdict = "BEATS persistence" if gain_persistence > 3 else (
            "ties persistence" if gain_persistence > -3 else "LOSES to persistence")
        print(f"{label:<22} {model_value:>9.4f} {constant_value:>10.4f} {persistence_value:>9.4f} "
              f"{gain_constant:>8.1f}% {gain_persistence:>10.1f}%  {verdict}")
        results[label] = {
            "model": model_value,
            "constant": constant_value,
            "persistence": persistence_value,
            "gain_vs_constant_pct": gain_constant,
            "gain_vs_persistence_pct": gain_persistence,
        }

    print(f"{'motion class acc':<22} {motion_accuracy:>9.4f} {majority_accuracy:>10.4f} "
          f"{'-':>9} {(motion_accuracy-majority_accuracy)*100:>8.1f}pp")
    results["motion"] = {"model": motion_accuracy, "majority": majority_accuracy}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-session", default="models/checkpoints/idr_v1_full/best_model.pt")
    parser.add_argument("--within-session", default="models/checkpoints/idr_within/best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, normalizer = build_split(verbose=False)
    train_split = splits["train"]

    summary = {}

    cross_path = Path(args.cross_session)
    if cross_path.exists():
        model, _ = load_model(cross_path, device)
        prediction = predict(model, splits["validation"], normalizer, device)
        summary["cross_session_validation"] = report(
            "CROSS-SESSION: held-out session S3a", train_split, splits["validation"], prediction
        )

    within_path = Path(args.within_session)
    if within_path.exists():
        within_train, within_validation = temporal_resplit(train_split)
        model, _ = load_model(within_path, device)
        prediction = predict(model, within_validation, normalizer, device)
        summary["within_session_validation"] = report(
            "WITHIN-SESSION: same drives, later in time", within_train, within_validation, prediction
        )

    output = Path("outputs/learnability.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("Beating 'constant' only shows the model noticed its inputs exist.")
    print("Beating 'persistence' is the bar that matters: a constant-velocity INS already")
    print("achieves it with no network, and VehicleFusionEkf.predictVelocity does exactly")
    print("that. A target where the model merely ties persistence is a target where the")
    print("network contributes nothing to dead reckoning.")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
