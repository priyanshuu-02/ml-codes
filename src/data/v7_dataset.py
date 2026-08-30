"""State-conditioned dataset used by the V7 dead-reckoning model.

The initial speed is a legitimate navigation state: it is supplied by the last
trusted GNSS update (and subsequently by the model during an outage).  It is
not a future label and makes velocity/displacement observable from IMU data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data.pytorch_dataset import load_split, get_normalizer
from src.data.targets import prepare_session, global_to_vehicle_frame


SAMPLE_RATE_HZ = 10.0


def build_v7_windows(imu, targets, window_size=20, stride=2):
    imu = np.asarray(imu, dtype=np.float32)
    if len(imu) < window_size:
        return None
    east = targets["position_east_m"].to_numpy(dtype=np.float64)
    north = targets["position_north_m"].to_numpy(dtype=np.float64)
    heading = targets["heading_deg"].to_numpy(dtype=np.float64)
    speed = targets["speed_mps"].to_numpy(dtype=np.float64)
    yaw = targets["yaw_rate_rad_s"].to_numpy(dtype=np.float64)
    motion = targets["motion_class"].to_numpy(dtype=np.int64)
    starts = np.arange(0, len(imu) - window_size + 1, stride, dtype=np.int64)
    ends = starts + window_size - 1
    delta_east, delta_north = east[ends] - east[starts], north[ends] - north[starts]
    forward, lateral = global_to_vehicle_frame(delta_east, delta_north, heading[starts])
    # Wrapped signed heading change across the complete window.  This is a
    # rollout-compatible target, unlike yaw-rate sampled only at its endpoint.
    heading_delta = np.radians((heading[ends] - heading[starts] + 180.0) % 360.0 - 180.0)
    return {
        "imu": np.stack([imu[s:s + window_size] for s in starts]).astype(np.float32),
        "speed": speed[ends].astype(np.float32),
        "initial_speed": speed[starts].astype(np.float32),
        "mean_speed": np.asarray([speed[s:e + 1].mean() for s, e in zip(starts, ends)], dtype=np.float32),
        "position": np.column_stack([forward, lateral]).astype(np.float32),
        "yaw_rate": yaw[ends].astype(np.float32),
        "heading_delta": heading_delta.astype(np.float32),
        "motion": motion[ends],
        "start": starts,
        "end": ends,
    }


class StateConditionedDataset(Dataset):
    def __init__(self, sessions, dataset, normalizer, window_size=20, stride=2):
        self.records = []
        for session_index, session in enumerate(sessions):
            imu, targets = prepare_session(dataset, session)
            windows = build_v7_windows(imu.to_numpy(), targets, window_size, stride)
            if windows is None:
                continue
            n = len(windows["imu"])
            windows["imu"] = normalizer.transform_imu(windows["imu"])
            for key in ("speed", "initial_speed", "mean_speed"):
                windows[key] = normalizer.transform_speed(windows[key])
            windows["position"] = normalizer.transform_position(windows["position"])
            windows["yaw_rate"] = normalizer.transform_yaw(windows["yaw_rate"])
            windows["session_index"] = np.full(n, session_index, dtype=np.int64)
            self.records.append(windows)
        if not self.records:
            raise RuntimeError("No valid V7 windows were created.")
        self.data = {key: np.concatenate([r[key] for r in self.records]) for key in self.records[0]}
        self.sessions = sessions
        if not np.isfinite(self.data["imu"]).all():
            raise ValueError("Non-finite V7 input found.")

    def __len__(self): return len(self.data["imu"])

    def __getitem__(self, index):
        return {
            "imu": torch.from_numpy(self.data["imu"][index]),
            "speed": torch.tensor(self.data["speed"][index], dtype=torch.float32),
            "initial_speed": torch.tensor(self.data["initial_speed"][index], dtype=torch.float32),
            "mean_speed": torch.tensor(self.data["mean_speed"][index], dtype=torch.float32),
            "position": torch.from_numpy(self.data["position"][index]),
            "yaw_rate": torch.tensor(self.data["yaw_rate"][index], dtype=torch.float32),
            "heading_delta": torch.tensor(self.data["heading_delta"][index], dtype=torch.float32),
            "motion": torch.tensor(self.data["motion"][index], dtype=torch.long),
            "session_index": torch.tensor(self.data["session_index"][index], dtype=torch.long),
            "start": torch.tensor(self.data["start"][index], dtype=torch.long),
            "end": torch.tensor(self.data["end"][index], dtype=torch.long),
        }


def create_v7_dataloader(split_name, batch_size=128, window_size=20, stride=2, balanced=False):
    normalizer = get_normalizer()
    dataset, sessions = load_split(split_name)
    pytorch_dataset = StateConditionedDataset(sessions, dataset, normalizer, window_size, stride)
    sampler = None
    shuffle = not balanced
    if balanced:
        labels = pytorch_dataset.data["motion"]
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
        shuffle = False
    return DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=0, pin_memory=torch.cuda.is_available())
