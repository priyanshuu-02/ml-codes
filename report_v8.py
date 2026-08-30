"""Detailed held-out V8 report: global, session, and motion-class metrics."""
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_v7 import predict_many
from src.data.pytorch_dataset import get_normalizer, load_split
from src.data.targets import prepare_session
from src.data.v7_dataset import build_v7_windows
from src.models.hybrid_v8 import V8DeadReckoningModel


def regression_metrics(speed_p, speed_t, pos_p, pos_t, heading_p, heading_t, motion_p, motion_t):
    error = pos_p - pos_t
    result = {
        "samples": int(len(speed_t)),
        "speed_mae_mps": float(np.abs(speed_p - speed_t).mean()),
        "speed_rmse_mps": float(np.sqrt(np.mean((speed_p - speed_t) ** 2))),
        "forward_mae_m": float(np.abs(error[:, 0]).mean()),
        "forward_rmse_m": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "lateral_mae_m": float(np.abs(error[:, 1]).mean()),
        "lateral_rmse_m": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "heading_delta_mae_rad": float(np.abs(heading_p - heading_t).mean()),
        "heading_delta_rmse_rad": float(np.sqrt(np.mean((heading_p - heading_t) ** 2))),
        "motion_accuracy": float(np.mean(motion_p == motion_t)),
    }
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("models/checkpoints/v8_heading_delta_run1/best_model.pt")
    model = V8DeadReckoningModel(input_channels=6, conv_dim=96, hidden_dim=128, dropout=0.15).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"]); model.eval()
    normalizer = get_normalizer(); dataset, sessions = load_split("test")
    all_parts, per_session = [], {}
    for session in sessions:
        imu, targets = prepare_session(dataset, session)
        windows = build_v7_windows(imu.to_numpy(dtype=np.float32), targets)
        speed_p, pos_p, heading_p, motion_p = predict_many(model, normalizer, windows["imu"], windows["initial_speed"], device)
        part = (speed_p, windows["speed"], pos_p, windows["position"], heading_p, windows["heading_delta"], motion_p, windows["motion"])
        all_parts.append(part); per_session[session["session_id"]] = regression_metrics(*part)
    merged = tuple(np.concatenate([part[i] for part in all_parts]) for i in range(8))
    truth, pred = merged[7], merged[6]
    classes = {str(i): {"samples": int((truth == i).sum()), "accuracy": float(np.mean(pred[truth == i] == i))} for i in range(3)}
    confusion = [[int(np.sum((truth == actual) & (pred == predicted))) for predicted in range(3)] for actual in range(3)]
    result = {"model": "V8 heading-delta", "global": regression_metrics(*merged), "per_session": per_session,
              "motion_class_metrics": classes, "motion_confusion_matrix": {"rows_actual_0_1_2": confusion}}
    output = Path("outputs/v8_heading_delta_run1/detailed_test_report.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
