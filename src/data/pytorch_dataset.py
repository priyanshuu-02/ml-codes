import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset
from src.data.targets import prepare_session
from src.data.windowing import create_windows
from src.data.normalization import TrainOnlyNormalizer


SPLIT_FILE = Path(
    "outputs/session_split.json"
)

NORMALIZATION_FILE = Path(
    "models/preprocessing/normalization.json"
)


class IOVNBDPyTorchDataset(Dataset):

    def __init__(
        self,
        sessions,
        dataset,
        window_size=20,
        stride=2,
        normalizer=None
    ):

        self.X = []
        self.y_speed = []
        self.y_position = []
        self.y_yaw_rate = []
        self.y_motion = []

        for session in sessions:

            imu, targets = prepare_session(
                dataset,
                session
            )

            X, speed, position, yaw_rate, motion = (
                create_windows(
                    imu.to_numpy(),
                    targets,
                    window_size=window_size,
                    stride=stride
                )
            )

            if len(X) == 0:
                continue

            self.X.append(X)
            self.y_speed.append(speed)
            self.y_position.append(position)
            self.y_yaw_rate.append(yaw_rate)
            self.y_motion.append(motion)

        if not self.X:

            raise RuntimeError(
                "No valid temporal windows were created."
            )

        self.X = np.concatenate(
            self.X,
            axis=0
        )

        self.y_speed = np.concatenate(
            self.y_speed,
            axis=0
        )

        self.y_position = np.concatenate(
            self.y_position,
            axis=0
        )

        self.y_yaw_rate = np.concatenate(
            self.y_yaw_rate,
            axis=0
        )

        self.y_motion = np.concatenate(
            self.y_motion,
            axis=0
        )

        # ----------------------------------------------------
        # Apply train-fitted normalization
        # ----------------------------------------------------

        if normalizer is not None:

            self.X = normalizer.transform_imu(
                self.X
            )

            self.y_speed = (
                normalizer.transform_speed(
                    self.y_speed
                )
            )

            self.y_position = (
                normalizer.transform_position(
                    self.y_position
                )
            )

            self.y_yaw_rate = (
                normalizer.transform_yaw(
                    self.y_yaw_rate
                )
            )

        # ----------------------------------------------------
        # Final safety checks
        # ----------------------------------------------------

        arrays = [
            self.X,
            self.y_speed,
            self.y_position,
            self.y_yaw_rate
        ]

        for array in arrays:

            if not np.isfinite(array).all():

                raise ValueError(
                    "NaN or Inf detected after normalization."
                )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return {
            "imu": torch.from_numpy(
                self.X[index].astype(
                    np.float32
                )
            ),

            "speed": torch.tensor(
                self.y_speed[index],
                dtype=torch.float32
            ),

            "position": torch.tensor(
                self.y_position[index],
                dtype=torch.float32
            ),

            "yaw_rate": torch.tensor(
                self.y_yaw_rate[index],
                dtype=torch.float32
            ),

            "motion": torch.tensor(
                self.y_motion[index],
                dtype=torch.long
            )
        }


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_name):

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        split = json.load(f)

    if split_name not in split:

        raise KeyError(
            f"Unknown split: {split_name}"
        )

    dataset = IOVNBDSynchronizedDataset(
        "data/IO-VNBD/synchronized"
    )

    sessions = dataset.get_sessions()

    selected_ids = set(
        split[split_name]
    )

    selected_sessions = [
        session
        for session in sessions
        if session["session_id"] in selected_ids
    ]

    return dataset, selected_sessions


# ============================================================
# FIT NORMALIZER ON TRAINING DATA ONLY
# ============================================================

def fit_training_normalizer():

    print(
        "\nFitting normalization using "
        "TRAINING DATA ONLY..."
    )

    dataset, sessions = load_split(
        "train"
    )

    all_X = []
    all_speed = []
    all_position = []
    all_yaw = []

    for session in sessions:

        imu, targets = prepare_session(
            dataset,
            session
        )

        X, speed, position, yaw_rate, motion = (
            create_windows(
                imu.to_numpy(),
                targets,
                window_size=20,
                stride=2
            )
        )

        if len(X) == 0:
            continue

        all_X.append(X)
        all_speed.append(speed)
        all_position.append(position)
        all_yaw.append(yaw_rate)

    X = np.concatenate(
        all_X,
        axis=0
    )

    speed = np.concatenate(
        all_speed,
        axis=0
    )

    position = np.concatenate(
        all_position,
        axis=0
    )

    yaw = np.concatenate(
        all_yaw,
        axis=0
    )

    normalizer = TrainOnlyNormalizer()

    normalizer.fit(
        X,
        speed,
        position,
        yaw
    )

    normalizer.save(
        NORMALIZATION_FILE
    )

    print(
        f"Normalization saved to:"
        f"\n{NORMALIZATION_FILE}"
    )

    return normalizer


# ============================================================
# LOAD OR CREATE NORMALIZER
# ============================================================

def get_normalizer():

    if NORMALIZATION_FILE.exists():

        return TrainOnlyNormalizer.load(
            NORMALIZATION_FILE
        )

    return fit_training_normalizer()


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(
    split_name,
    batch_size=128,
    window_size=20,
    stride=2,
    shuffle=False
):

    normalizer = get_normalizer()

    dataset, sessions = load_split(
        split_name
    )

    pytorch_dataset = IOVNBDPyTorchDataset(
        sessions=sessions,
        dataset=dataset,
        window_size=window_size,
        stride=stride,
        normalizer=normalizer
    )

    loader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NORMALIZED PYTORCH DATASET TEST")
    print("=" * 70)

    train_loader = create_dataloader(
        split_name="train",
        batch_size=128,
        window_size=20,
        stride=2,
        shuffle=True
    )

    val_loader = create_dataloader(
        split_name="validation",
        batch_size=128,
        window_size=20,
        stride=2,
        shuffle=False
    )

    train_batch = next(
        iter(train_loader)
    )

    val_batch = next(
        iter(val_loader)
    )

    print(
        f"\nTraining samples: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset):,}"
    )

    print("\nTraining batch:")

    for key, value in train_batch.items():

        print(
            f"{key}: "
            f"{tuple(value.shape)}"
        )

    print("\nNormalized training statistics:")

    print(
        f"IMU mean: "
        f"{train_batch['imu'].mean().item():.4f}"
    )

    print(
        f"IMU std: "
        f"{train_batch['imu'].std().item():.4f}"
    )

    print(
        f"Speed mean: "
        f"{train_batch['speed'].mean().item():.4f}"
    )

    print(
        f"Position mean: "
        f"{train_batch['position'].mean().item():.4f}"
    )

    print(
        f"Yaw mean: "
        f"{train_batch['yaw_rate'].mean().item():.4f}"
    )

    print("\nNaN/Inf checks:")

    for name, batch in [
        ("TRAIN", train_batch),
        ("VALIDATION", val_batch)
    ]:

        for key, value in batch.items():

            if torch.is_floating_point(value):

                if not torch.isfinite(value).all():

                    raise ValueError(
                        f"{name} {key} contains "
                        "NaN or Inf."
                    )

    print(
        "TRAIN: PASSED"
    )

    print(
        "VALIDATION: PASSED"
    )

    print(
        "\nNORMALIZED PYTORCH DATASET SUCCESSFUL"
    )