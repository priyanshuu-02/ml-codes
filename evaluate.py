import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.pytorch_dataset import (
    IOVNBDPyTorchDataset,
    load_split,
    get_normalizer
)

from src.models.hybrid import (
    IntelligentDeadReckoningModel
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 128

WINDOW_SIZE = 20
STRIDE = 2

CHECKPOINT = Path(
    "models/checkpoints/best_model.pt"
)

OUTPUT_DIR = Path(
    "outputs/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 80)
print("INTELLIGENT DEAD RECKONING — FINAL TEST EVALUATION")
print("=" * 80)

print(
    f"\nDevice: {DEVICE}"
)

dataset, test_sessions = load_split(
    "test"
)

print(
    f"Test sessions: "
    f"{len(test_sessions)}"
)

print("\nTest sessions:")

for session in test_sessions:

    print(
        f"  {session['session_id']} | "
        f"{session['category']}"
    )


# ============================================================
# DATASET
# ============================================================

normalizer = get_normalizer()

test_dataset = IOVNBDPyTorchDataset(
    sessions=test_sessions,
    dataset=dataset,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    normalizer=normalizer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(
    f"\nTest samples: "
    f"{len(test_dataset):,}"
)


# ============================================================
# MODEL
# ============================================================

model = IntelligentDeadReckoningModel(
    input_channels=6,
    conv_dim=96,
    hidden_dim=128,
    dropout=0.1
).to(DEVICE)


# ============================================================
# LOAD BEST MODEL
# ============================================================

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Best model not found:\n{CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"\nLoaded best model:"
    f"\n{CHECKPOINT}"
)

print(
    f"Best validation loss: "
    f"{checkpoint.get('best_val_loss', 'N/A')}"
)


# ============================================================
# EVALUATION
# ============================================================

all_speed_pred = []
all_speed_true = []

all_position_pred = []
all_position_true = []

all_yaw_pred = []
all_yaw_true = []

all_motion_pred = []
all_motion_true = []


print("\nRunning test evaluation...")

with torch.no_grad():

    for batch in test_loader:

        imu = batch["imu"].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(imu)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        speed_pred = outputs[
            "speed"
        ]

        position_pred = outputs[
            "position"
        ]

        yaw_pred = outputs[
            "yaw_rate"
        ]

        motion_logits = outputs[
            "motion_logits"
        ]

        motion_pred = torch.argmax(
            motion_logits,
            dim=1
        )

        # ----------------------------------------------------
        # Store normalized values
        # ----------------------------------------------------

        all_speed_pred.append(
            speed_pred.cpu().numpy()
        )

        all_speed_true.append(
            batch["speed"].numpy()
        )

        all_position_pred.append(
            position_pred.cpu().numpy()
        )

        all_position_true.append(
            batch["position"].numpy()
        )

        all_yaw_pred.append(
            yaw_pred.cpu().numpy()
        )

        all_yaw_true.append(
            batch["yaw_rate"].numpy()
        )

        all_motion_pred.append(
            motion_pred.cpu().numpy()
        )

        all_motion_true.append(
            batch["motion"].numpy()
        )


# ============================================================
# CONCATENATE
# ============================================================

speed_pred = np.concatenate(
    all_speed_pred
)

speed_true = np.concatenate(
    all_speed_true
)

position_pred = np.concatenate(
    all_position_pred
)

position_true = np.concatenate(
    all_position_true
)

yaw_pred = np.concatenate(
    all_yaw_pred
)

yaw_true = np.concatenate(
    all_yaw_true
)

motion_pred = np.concatenate(
    all_motion_pred
)

motion_true = np.concatenate(
    all_motion_true
)


# ============================================================
# CONVERT NORMALIZED PREDICTIONS BACK TO PHYSICAL UNITS
# ============================================================

speed_pred = normalizer.inverse_speed(
    speed_pred
)

speed_true = normalizer.inverse_speed(
    speed_true
)

position_pred = normalizer.inverse_position(
    position_pred
)

position_true = normalizer.inverse_position(
    position_true
)

yaw_pred = normalizer.inverse_yaw(
    yaw_pred
)

yaw_true = normalizer.inverse_yaw(
    yaw_true
)


# ============================================================
# METRICS
# ============================================================

speed_error = (
    speed_pred - speed_true
)

position_error = (
    position_pred - position_true
)

yaw_error = (
    yaw_pred - yaw_true
)


speed_mae = np.mean(
    np.abs(speed_error)
)

speed_rmse = np.sqrt(
    np.mean(
        speed_error ** 2
    )
)


forward_mae = np.mean(
    np.abs(
        position_error[:, 0]
    )
)

lateral_mae = np.mean(
    np.abs(
        position_error[:, 1]
    )
)

position_rmse = np.sqrt(
    np.mean(
        position_error ** 2
    )
)

yaw_mae = np.mean(
    np.abs(yaw_error)
)

yaw_rmse = np.sqrt(
    np.mean(
        yaw_error ** 2
    )
)


# ============================================================
# MOTION CLASSIFICATION
# ============================================================

motion_accuracy = np.mean(
    motion_pred == motion_true
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

class_metrics = {}

for class_id in [0, 1, 2]:

    mask = (
        motion_true == class_id
    )

    if mask.sum() == 0:

        class_metrics[
            str(class_id)
        ] = {
            "samples": 0,
            "accuracy": None
        }

    else:

        class_metrics[
            str(class_id)
        ] = {

            "samples": int(
                mask.sum()
            ),

            "accuracy": float(
                np.mean(
                    motion_pred[mask]
                    ==
                    motion_true[mask]
                )
            )
        }


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    f"\nSpeed MAE: "
    f"{speed_mae:.4f} m/s"
)

print(
    f"Speed RMSE: "
    f"{speed_rmse:.4f} m/s"
)

print(
    f"\nForward displacement MAE: "
    f"{forward_mae:.4f} m"
)

print(
    f"Lateral displacement MAE: "
    f"{lateral_mae:.4f} m"
)

print(
    f"2D displacement RMSE: "
    f"{position_rmse:.4f} m"
)

print(
    f"\nYaw-rate MAE: "
    f"{yaw_mae:.6f} rad/s"
)

print(
    f"Yaw-rate RMSE: "
    f"{yaw_rmse:.6f} rad/s"
)

print(
    f"\nMotion accuracy: "
    f"{motion_accuracy * 100:.2f}%"
)

print("\nMotion classes:")

for class_id, metrics in class_metrics.items():

    print(
        f"  Class {class_id}: "
        f"{metrics['samples']} samples, "
        f"accuracy="
        f"{metrics['accuracy']}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "checkpoint": str(
        CHECKPOINT
    ),

    "best_validation_loss":
        checkpoint.get(
            "best_val_loss"
        ),

    "test_samples":
        len(test_dataset),

    "speed_mae_mps":
        float(speed_mae),

    "speed_rmse_mps":
        float(speed_rmse),

    "forward_mae_m":
        float(forward_mae),

    "lateral_mae_m":
        float(lateral_mae),

    "position_rmse_m":
        float(position_rmse),

    "yaw_mae_rad_s":
        float(yaw_mae),

    "yaw_rmse_rad_s":
        float(yaw_rmse),

    "motion_accuracy":
        float(motion_accuracy),

    "motion_class_metrics":
        class_metrics
}


results_file = (
    OUTPUT_DIR /
    "test_results.json"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    f"\nResults saved:"
    f"\n{results_file}"
)

print(
    "\nFINAL TEST EVALUATION COMPLETE"
)