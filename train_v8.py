"""Train V8: state-conditioned displacement plus integrated heading change."""
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
from src.losses.v8_loss import V8Loss
from src.models.hybrid_v8 import V8DeadReckoningModel


def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train); total = 0.0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch in tqdm(loader, desc="train" if train else "validation", leave=False):
            imu = batch["imu"].to(device, non_blocking=True)
            state = batch["initial_speed"].to(device, non_blocking=True)
            targets = {key: batch[key].to(device, non_blocking=True) for key in ("speed", "position", "heading_delta", "motion")}
            if train: optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = criterion(model(imu, state), targets)["total"]
            if not torch.isfinite(loss): raise RuntimeError("Non-finite V8 loss")
            if train:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="v8_heading_delta")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir, out_dir = Path("models/checkpoints") / args.experiment, Path("outputs") / args.experiment
    ckpt_dir.mkdir(parents=True, exist_ok=False); out_dir.mkdir(parents=True, exist_ok=False)
    train_loader = create_v7_dataloader("train", args.batch_size, balanced=True)
    val_loader = create_v7_dataloader("validation", args.batch_size, balanced=False)
    counts = np.bincount(train_loader.dataset.data["motion"], minlength=3)
    class_weights = np.minimum(np.sqrt(counts.max() / np.maximum(counts, 1)), 1.5)
    model = V8DeadReckoningModel(input_channels=6, conv_dim=96, hidden_dim=128, dropout=0.15).to(device)
    criterion = V8Loss(class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    config = {"architecture": "V8 state-conditioned with heading-delta target", "seed": 42, "test_set_used_for_selection": False,
              "class_counts": counts.tolist(), "class_weights": class_weights.tolist()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, stalled, history, started = float("inf"), 0, [], time.time()
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, scaler, False)
        scheduler.step(val_loss); improved = val_loss < best
        best, stalled = (val_loss, 0) if improved else (best, stalled + 1)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_loss": best, "config": config}
        torch.save(checkpoint, ckpt_dir / "last_checkpoint.pt")
        if improved: torch.save(checkpoint, ckpt_dir / "best_model.pt")
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
               "learning_rate": optimizer.param_groups[0]["lr"], "best": improved}
        history.append(row); (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if stalled >= 8: print("Early stopping.", flush=True); break
    print(f"Complete in {(time.time()-started)/3600:.2f} h; best validation loss={best:.6f}", flush=True)


if __name__ == "__main__": main()
