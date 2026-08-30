"""Train V7 without altering the V1 checkpoint or its preprocessing artefacts."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.v7_dataset import create_v7_dataloader
from src.losses.v7_loss import V7Loss
from src.models.hybrid_v7 import V7DeadReckoningModel


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def targets_from_batch(batch, device):
    keys = ("speed", "position", "yaw_rate", "motion")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train)
    total, batches = 0.0, 0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch in tqdm(loader, desc="train" if train else "validation", leave=False):
            imu = batch["imu"].to(device, non_blocking=True)
            initial_speed = batch["initial_speed"].to(device, non_blocking=True)
            targets = targets_from_batch(batch, device)
            if train: optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                losses = criterion(model(imu, initial_speed), targets)
            if not torch.isfinite(losses["total"]): raise RuntimeError("Non-finite V7 loss")
            if train:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            total += losses["total"].item(); batches += 1
    return total / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="v7_state_turn_aware")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("models/checkpoints") / args.experiment
    output_dir = Path("outputs") / args.experiment
    run_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_loader = create_v7_dataloader("train", args.batch_size, balanced=True)
    val_loader = create_v7_dataloader("validation", args.batch_size, balanced=False)
    class_counts = np.bincount(train_loader.dataset.data["motion"], minlength=3)
    # Mild weighting complements sampling, without allowing rare turns to dominate.
    class_weights = np.sqrt(class_counts.max() / np.maximum(class_counts, 1))
    class_weights = np.minimum(class_weights, 1.5)
    model = V7DeadReckoningModel(input_channels=6, conv_dim=96, hidden_dim=128, dropout=0.15).to(device)
    criterion = V7Loss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best, stalled, history = float("inf"), 0, []
    config = {"experiment": args.experiment, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
              "architecture": "V7 state-conditioned ConvNeXt-1D + GRU + PatchTST", "class_counts": class_counts.tolist(),
              "class_weights": class_weights.tolist(), "test_set_used_for_selection": False}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.time()
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
        scheduler.step(val_loss)
        improved = val_loss < best
        if improved: best, stalled = val_loss, 0
        else: stalled += 1
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": best, "config": config}
        torch.save(checkpoint, run_dir / "last_checkpoint.pt")
        if improved: torch.save(checkpoint, run_dir / "best_model.pt")
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
               "learning_rate": optimizer.param_groups[0]["lr"], "best": improved}
        history.append(row); (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row))
        if stalled >= 8:
            print("Early stopping."); break
    print(f"Complete in {(time.time()-started)/3600:.2f} h; best validation loss={best:.6f}")


if __name__ == "__main__": main()
