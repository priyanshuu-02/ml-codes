import json
import time
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.pytorch_dataset import create_dataloader
from src.models.hybrid import IntelligentDeadReckoningModel
from src.losses.multitask_loss import MultiTaskLoss


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128
MAX_EPOCHS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.1

WINDOW_SIZE = 20
STRIDE = 2

EARLY_STOPPING_PATIENCE = 8

GRADIENT_CLIP = 1.0

CHECKPOINT_DIR = Path(
    "models/checkpoints"
)

OUTPUT_DIR = Path(
    "outputs"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LAST_CHECKPOINT = (
    CHECKPOINT_DIR /
    "last_checkpoint.pt"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR /
    "best_model.pt"
)

HISTORY_FILE = (
    OUTPUT_DIR /
    "training_history.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

print("=" * 80)
print("INTELLIGENT DEAD RECKONING TRAINING")
print("=" * 80)

print(
    f"\nDevice: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# DATA
# ============================================================

print("\nLoading normalized training data...")

train_loader = create_dataloader(
    split_name="train",
    batch_size=BATCH_SIZE,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    shuffle=True
)

print(
    f"Training samples: "
    f"{len(train_loader.dataset):,}"
)


print("\nLoading normalized validation data...")

val_loader = create_dataloader(
    split_name="validation",
    batch_size=BATCH_SIZE,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    shuffle=False
)

print(
    f"Validation samples: "
    f"{len(val_loader.dataset):,}"
)


# ============================================================
# MODEL
# ============================================================

model = IntelligentDeadReckoningModel(
    input_channels=6,
    conv_dim=96,
    hidden_dim=128,
    dropout=DROPOUT
).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = MultiTaskLoss(
    speed_weight=1.0,
    position_weight=2.0,
    yaw_weight=1.0,
    motion_weight=0.5
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)


# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = DEVICE.type == "cuda"

scaler = GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ============================================================
# RESUME STATE
# ============================================================

start_epoch = 0

best_val_loss = float("inf")

epochs_without_improvement = 0

history = []


# ============================================================
# RESUME FROM LAST CHECKPOINT
# ============================================================

if LAST_CHECKPOINT.exists():

    print(
        f"\nCheckpoint found:"
        f"\n{LAST_CHECKPOINT}"
    )

    checkpoint = torch.load(
        LAST_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    epochs_without_improvement = checkpoint[
        "epochs_without_improvement"
    ]

    history = checkpoint.get(
        "history",
        []
    )

    print(
        f"Resuming from epoch "
        f"{start_epoch + 1}"
    )


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_batch(batch):

    for name, tensor in batch.items():

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"Non-finite values detected "
                f"in target/input: {name}"
            )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    batches = 0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for batch in progress:

        validate_batch(batch)

        imu = batch["imu"].to(
            DEVICE,
            non_blocking=True
        )

        targets = {

            "speed": batch["speed"].to(
                DEVICE,
                non_blocking=True
            ),

            "position": batch["position"].to(
                DEVICE,
                non_blocking=True
            ),

            "yaw_rate": batch["yaw_rate"].to(
                DEVICE,
                non_blocking=True
            ),

            "motion": batch["motion"].to(
                DEVICE,
                non_blocking=True
            )
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            outputs = model(imu)

            losses = criterion(
                outputs,
                targets
            )

            loss = losses["total"]

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite training loss detected."
            )

        scaler.scale(loss).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += loss.item()

        batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / batches


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    batches = 0

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False
    )

    for batch in progress:

        validate_batch(batch)

        imu = batch["imu"].to(
            DEVICE,
            non_blocking=True
        )

        targets = {

            "speed": batch["speed"].to(
                DEVICE,
                non_blocking=True
            ),

            "position": batch["position"].to(
                DEVICE,
                non_blocking=True
            ),

            "yaw_rate": batch["yaw_rate"].to(
                DEVICE,
                non_blocking=True
            ),

            "motion": batch["motion"].to(
                DEVICE,
                non_blocking=True
            )
        }

        with autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            outputs = model(imu)

            losses = criterion(
                outputs,
                targets
            )

            loss = losses["total"]

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite validation loss detected."
            )

        total_loss += loss.item()

        batches += 1

    return total_loss / batches


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

training_start = time.time()


for epoch in range(
    start_epoch,
    MAX_EPOCHS
):

    epoch_start = time.time()

    print(
        f"\nEpoch "
        f"{epoch + 1}/{MAX_EPOCHS}"
    )

    train_loss = train_one_epoch()

    val_loss = validate()

    scheduler.step(
        val_loss
    )

    learning_rate = optimizer.param_groups[0][
        "lr"
    ]

    epoch_time = (
        time.time()
        - epoch_start
    )

    improved = (
        val_loss < best_val_loss
    )

    if improved:

        best_val_loss = val_loss

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "validation_loss": val_loss,

        "learning_rate": learning_rate,

        "epoch_time_seconds": epoch_time,

        "best": improved
    })


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"\nTrain Loss: "
        f"{train_loss:.6f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.6f}"
    )

    print(
        f"Learning Rate: "
        f"{learning_rate:.8f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time / 60:.2f} min"
    )


    # --------------------------------------------------------
    # SAVE CHECKPOINT EVERY EPOCH
    # --------------------------------------------------------

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_val_loss":
            best_val_loss,

        "epochs_without_improvement":
            epochs_without_improvement,

        "history":
            history,

        "config": {

            "input_channels": 6,

            "conv_dim": 96,

            "hidden_dim": 128,

            "dropout": DROPOUT,

            "batch_size": BATCH_SIZE,

            "learning_rate": LEARNING_RATE,

            "weight_decay": WEIGHT_DECAY,

            "window_size": WINDOW_SIZE,

            "stride": STRIDE
        }
    }

    torch.save(
        checkpoint,
        LAST_CHECKPOINT
    )

    print(
        "Checkpoint saved:"
        f" {LAST_CHECKPOINT}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if improved:

        torch.save(
            checkpoint,
            BEST_CHECKPOINT
        )

        print(
            "*** NEW BEST MODEL ***"
        )

        print(
            f"Best validation loss:"
            f" {best_val_loss:.6f}"
        )


    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        print(
            f"No validation improvement "
            f"for {EARLY_STOPPING_PATIENCE} epochs."
        )

        break


# ============================================================
# COMPLETE
# ============================================================

total_time = (
    time.time()
    - training_start
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"\nBest validation loss:"
    f" {best_val_loss:.6f}"
)

print(
    f"Total training time:"
    f" {total_time / 3600:.2f} hours"
)

print(
    f"\nBest model:"
    f"\n{BEST_CHECKPOINT}"
)

print(
    f"\nLatest checkpoint:"
    f"\n{LAST_CHECKPOINT}"
)

print(
    f"\nTraining history:"
    f"\n{HISTORY_FILE}"
)