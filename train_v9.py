"""Closed-loop sequence training for V9 (five non-overlapping 2-second windows)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.v9_dataset import create_v9_dataloader
from src.losses.v9_loss import V9Loss
from src.models.hybrid_v8 import V8DeadReckoningModel


def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train); total = 0.0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch in tqdm(loader, desc="train" if train else "validation", leave=False):
            imu, state = batch["imu"].to(device), batch["initial_speed"].to(device)
            targets = {key: batch[key].to(device) for key in ("speed", "position", "heading_delta", "motion")}
            if train: optimizer.zero_grad(set_to_none=True)
            outputs = []
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                # Predicted speed, not target speed, becomes the next navigation state.
                for step in range(imu.shape[1]):
                    output = model(imu[:, step], state); outputs.append(output); state = output["speed"]
                loss = criterion(outputs, targets)["total"]
            if not torch.isfinite(loss): raise RuntimeError("Non-finite V9 loss")
            if train:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", default="v9_closed_loop_run1")
    parser.add_argument("--epochs", type=int, default=35); parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir, output_dir = Path("models/checkpoints") / args.experiment, Path("outputs") / args.experiment
    ckpt_dir.mkdir(parents=True, exist_ok=False); output_dir.mkdir(parents=True, exist_ok=False)
    train_loader, normalizer = create_v9_dataloader("train", args.batch_size, shuffle=True)
    val_loader, _ = create_v9_dataloader("validation", args.batch_size)
    labels = np.concatenate([r["motion"] for r in train_loader.dataset.records]); counts = np.bincount(labels, minlength=3)
    weights = np.minimum(np.sqrt(counts.max() / np.maximum(counts, 1)), 1.5)
    model = V8DeadReckoningModel(input_channels=6, conv_dim=96, hidden_dim=128, dropout=0.15).to(device)
    criterion = V9Loss(weights, normalizer.position_mean, normalizer.position_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    config = {"architecture": "V9 V8 encoder + predicted-state sequence unroll", "sequence_length": 5,
              "sequence_seconds": 9.5, "trajectory_weight": 0.30, "test_set_used_for_selection": False}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, stalled, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
        scheduler.step(val_loss); improved = val_loss < best
        best, stalled = (val_loss, 0) if improved else (best, stalled + 1)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_loss": best, "config": config}
        torch.save(checkpoint, ckpt_dir / "last_checkpoint.pt")
        if improved: torch.save(checkpoint, ckpt_dir / "best_model.pt")
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"], "best": improved}
        history.append(row); (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if stalled >= 8: print("Early stopping.", flush=True); break


if __name__ == "__main__": main()
