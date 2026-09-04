"""
Train the IDR-V1 motion model.

Checkpoint selection uses validation forward-displacement and heading error, because those
are the two quantities that actually drive trajectory integration. Selecting on total loss
would let a model win by becoming confidently vague, and selecting on the held-out test
session would invalidate the only generalisation evidence available.

The test session is never read here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import build_split, check_plausibility
from src.models.idr_v1 import IdrV1Loss, IdrV1Model


def to_tensors(split, normalizer):
    """Normalise a split and wrap it as tensors, in the contract's units."""
    imu = torch.from_numpy(normalizer.imu(split["imu"]).astype(np.float32))
    initial_speed = torch.from_numpy(
        normalizer.speed_forward(split["initial_speed"]).astype(np.float32)
    )
    speed = torch.from_numpy(normalizer.speed_forward(split["speed"]).astype(np.float32))
    acceleration = torch.from_numpy(
        normalizer.acceleration_forward(split["acceleration"]).astype(np.float32)
    )
    position = torch.from_numpy(normalizer.position_forward(split["position"]).astype(np.float32))
    # Heading delta stays in raw radians: it is small, already zero-centred, and the runtime
    # consumes it directly without a denormalisation step.
    heading_delta = torch.from_numpy(split["heading_delta"].astype(np.float32))
    motion = torch.from_numpy(split["motion"].astype(np.int64))
    return TensorDataset(imu, initial_speed, speed, acceleration, position, heading_delta, motion)


def unpack(batch, device, ablate_imu=False):
    imu, initial_speed, speed, acceleration, position, heading_delta, motion = [
        item.to(device, non_blocking=True) for item in batch
    ]
    if ablate_imu:
        imu = torch.zeros_like(imu)
    targets = {
        "speed": speed,
        "acceleration": acceleration,
        "position": position,
        "heading_delta": heading_delta,
        "motion": motion,
    }
    return imu, initial_speed, targets


def temporal_resplit(split, train_fraction=0.7, gap_windows=40):
    """
    Split each session temporally rather than holding out whole sessions.

    A gap is dropped either side of the boundary because windows overlap by 90 percent at
    stride 2: without it the first validation windows would share most of their samples
    with the last training windows, which would leak and flatter the result.
    """
    keys = [key for key in split if key != "session_id"]
    train_parts, validation_parts = [], []

    for session_id in sorted(set(split["session_id"].tolist())):
        mask = split["session_id"] == session_id
        indices = np.where(mask)[0]
        order = indices[np.argsort(split["start"][indices])]
        cut = int(len(order) * train_fraction)
        train_indices = order[: max(cut - gap_windows, 0)]
        validation_indices = order[cut + gap_windows:]
        if len(train_indices) == 0 or len(validation_indices) == 0:
            continue
        train_parts.append(train_indices)
        validation_parts.append(validation_indices)

    def gather(parts):
        selected = np.concatenate(parts)
        result = {key: split[key][selected] for key in keys}
        result["session_id"] = split["session_id"][selected]
        return result

    return gather(train_parts), gather(validation_parts)


def constant_baselines(train_split, validation_split):
    """
    What a model achieves by ignoring its inputs and emitting the training mean.

    Any metric that fails to beat this is not evidence of learning. Reporting it alongside
    the model is what makes the difference between a real result and a flattering one.
    """
    return {
        "forward_mae_m": float(np.abs(
            validation_split["position"][:, 0] - train_split["position"][:, 0].mean()
        ).mean()),
        "lateral_mae_m": float(np.abs(
            validation_split["position"][:, 1] - train_split["position"][:, 1].mean()
        ).mean()),
        "heading_mae_rad": float(np.abs(
            validation_split["heading_delta"] - train_split["heading_delta"].mean()
        ).mean()),
        "speed_mae_mps": float(np.abs(
            validation_split["speed"] - train_split["speed"].mean()
        ).mean()),
        "acceleration_mae_mps2": float(np.abs(
            validation_split["acceleration"] - train_split["acceleration"].mean()
        ).mean()),
    }


def persistence_baselines(validation_split):
    """
    The other baseline that matters: assume nothing changed during the window.

    Forward displacement becomes initial_speed times the span, speed stays at
    initial_speed, and heading does not turn. Speed is strongly autocorrelated over 1.9 s,
    so this is a demanding bar for the speed and forward heads and a trivial one for
    heading. Beating it on heading is the clearest sign the gyro is being used.
    """
    from src.data.idr_dataset import WINDOW_SPAN_SECONDS

    return {
        "forward_mae_m": float(np.abs(
            validation_split["position"][:, 0]
            - validation_split["initial_speed"] * WINDOW_SPAN_SECONDS
        ).mean()),
        "lateral_mae_m": float(np.abs(validation_split["position"][:, 1]).mean()),
        "heading_mae_rad": float(np.abs(validation_split["heading_delta"]).mean()),
        "speed_mae_mps": float(np.abs(
            validation_split["speed"] - validation_split["initial_speed"]
        ).mean()),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, normalizer, device, ablate_imu=False):
    """Validation metrics reported in real units, not normalised ones."""
    model.eval()
    total_loss = 0.0
    samples = 0
    forward_error = []
    lateral_error = []
    heading_error = []
    speed_error = []
    acceleration_error = []
    correct = 0

    for batch in loader:
        imu, initial_speed, targets = unpack(batch, device, ablate_imu)
        outputs = model(imu, initial_speed)
        losses = criterion(outputs, targets)

        count = imu.shape[0]
        total_loss += losses["total"].item() * count
        samples += count

        position = outputs[4].cpu().numpy()
        truth_position = targets["position"].cpu().numpy()
        position_metres = normalizer.position_inverse(position)
        truth_metres = normalizer.position_inverse(truth_position)
        forward_error.append(np.abs(position_metres[:, 0] - truth_metres[:, 0]))
        lateral_error.append(np.abs(position_metres[:, 1] - truth_metres[:, 1]))

        heading_error.append(np.abs(outputs[6].cpu().numpy() - targets["heading_delta"].cpu().numpy()))

        speed_metres = normalizer.speed_inverse(outputs[0].cpu().numpy())
        truth_speed = normalizer.speed_inverse(targets["speed"].cpu().numpy())
        speed_error.append(np.abs(speed_metres - truth_speed))

        acceleration_real = normalizer.acceleration_inverse(outputs[2].cpu().numpy())
        truth_acceleration = normalizer.acceleration_inverse(targets["acceleration"].cpu().numpy())
        acceleration_error.append(np.abs(acceleration_real - truth_acceleration))

        correct += int((outputs[8].argmax(dim=1) == targets["motion"]).sum())

    return {
        "loss": total_loss / max(samples, 1),
        "forward_mae_m": float(np.concatenate(forward_error).mean()),
        "lateral_mae_m": float(np.concatenate(lateral_error).mean()),
        "heading_mae_rad": float(np.concatenate(heading_error).mean()),
        "speed_mae_mps": float(np.concatenate(speed_error).mean()),
        "acceleration_mae_mps2": float(np.concatenate(acceleration_error).mean()),
        "motion_accuracy": correct / max(samples, 1),
    }


def selection_score(metrics):
    """
    Lower is better. Combines the two errors that compound through integration:
    forward displacement in metres, and heading in radians scaled to a comparable size.
    """
    return metrics["forward_mae_m"] + 20.0 * metrics["heading_mae_rad"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="idr_v1_run1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced", action="store_true",
                        help="Sample motion classes evenly instead of by natural frequency.")
    parser.add_argument("--include-borderline", action="store_true",
                        help="Add S3b and S4, whose IMU correspondence is weaker.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ablate-imu",
        action="store_true",
        help="Zero the IMU input so only initial_speed remains. This is the control that "
             "reveals how much the model actually reads the sensors: V8 scored acceptable "
             "per-window speed error while learning almost nothing from the IMU, because "
             "speed is autocorrelated and initial_speed was handed to it.",
    )
    parser.add_argument(
        "--within-session",
        action="store_true",
        help="Split each training session temporally instead of holding out whole sessions. "
             "Diagnostic only: it isolates whether the IMU carries usable signal at all "
             "from whether that signal transfers across phone mountings. Each session has "
             "its own unknown mount orientation, so 'gyro_ch1' does not mean the same thing "
             "in two different sessions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    print("\nbuilding split")
    splits, normalizer = build_split(include_borderline=args.include_borderline, verbose=True)

    train_split = splits["train"]
    validation_split = splits["validation"]

    if args.within_session:
        train_split, validation_split = temporal_resplit(train_split)
        print("\nwithin-session diagnostic split")
        print(f"  train      {len(train_split['imu']):,} windows")
        print(f"  validation {len(validation_split['imu']):,} windows")

    if validation_split is None:
        raise RuntimeError("Validation split is empty.")

    # Refuse to train on a physically impossible target set. This is the check that was
    # missing when the previous artifact shipped.
    reasons = check_plausibility(
        train_split["speed"], train_split["position"][:, 0], train_split["position"][:, 1]
    )
    if reasons:
        raise RuntimeError("Training targets failed the plausibility gate: " + "; ".join(reasons))
    print("\nplausibility gate: PASSED")

    train_dataset = to_tensors(train_split, normalizer)
    validation_dataset = to_tensors(validation_split, normalizer)

    sampler = None
    shuffle = True
    if args.balanced:
        counts = np.bincount(train_split["motion"], minlength=3).astype(np.float64)
        weights = 1.0 / counts[train_split["motion"]]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=512, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    constant = constant_baselines(train_split, validation_split)
    persistence = persistence_baselines(validation_split)
    print("\nvalidation baselines the model must beat to count as learning")
    print(f"{'':<16} {'constant':>10} {'persistence':>12}")
    for key in ("forward_mae_m", "lateral_mae_m", "heading_mae_rad", "speed_mae_mps"):
        constant_value = constant[key]
        persistence_value = persistence.get(key, float("nan"))
        print(f"  {key:<14} {constant_value:>10.4f} {persistence_value:>12.4f}")

    model = IdrV1Model(dropout=args.dropout).to(device)
    criterion = IdrV1Loss().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.learning_rate, total_steps=args.epochs * len(train_loader),
        pct_start=0.25,
    )

    print(f"\nparameters      : {model.parameter_count():,}")
    print(f"train windows   : {len(train_dataset):,}")
    print(f"val windows     : {len(validation_dataset):,}")
    print(f"batches / epoch : {len(train_loader):,}")

    checkpoint_dir = Path("models/checkpoints") / args.experiment
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path("outputs") / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(output_dir / "normalization.json")

    history = []
    best_score = float("inf")
    best_epoch = -1

    print("\n" + "=" * 96)
    print(f"{'ep':>3} {'train':>9} {'val':>9} {'fwdMAE':>8} {'latMAE':>8} "
          f"{'hdgMAE':>8} {'spdMAE':>8} {'accMAE':>8} {'motAcc':>7} {'score':>8} {'s':>5}")
    print("=" * 96)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        for batch in train_loader:
            imu, initial_speed, targets = unpack(batch, device, args.ablate_imu)
            optimiser.zero_grad(set_to_none=True)
            outputs = model(imu, initial_speed)
            losses = criterion(outputs, targets)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()
            running += losses["total"].item() * imu.shape[0]
            seen += imu.shape[0]

        train_loss = running / max(seen, 1)
        metrics = evaluate(model, validation_loader, criterion, normalizer, device, args.ablate_imu)
        score = selection_score(metrics)
        elapsed = time.time() - started

        print(f"{epoch:>3} {train_loss:>9.4f} {metrics['loss']:>9.4f} "
              f"{metrics['forward_mae_m']:>8.3f} {metrics['lateral_mae_m']:>8.3f} "
              f"{metrics['heading_mae_rad']:>8.4f} {metrics['speed_mae_mps']:>8.3f} "
              f"{metrics['acceleration_mae_mps2']:>8.3f} {metrics['motion_accuracy']:>7.3f} "
              f"{score:>8.3f} {elapsed:>5.1f}"
              + ("  *" if score < best_score else ""))

        history.append({"epoch": epoch, "train_loss": train_loss, "score": score, **metrics})

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "selection_score": score,
                    "preprocessing_version": "idr-v1",
                    "args": vars(args),
                },
                checkpoint_dir / "best_model.pt",
            )

    (output_dir / "training_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "ablate_imu": args.ablate_imu,
                "baselines": {"constant": constant, "persistence": persistence},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 96)
    print(f"best epoch {best_epoch} with selection score {best_score:.3f}")
    print(f"checkpoint : {checkpoint_dir / 'best_model.pt'}")
    print(f"history    : {output_dir / 'training_history.json'}")
    print("\nThe test session was not read during training.")


if __name__ == "__main__":
    main()
