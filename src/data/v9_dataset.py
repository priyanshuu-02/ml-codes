"""Contiguous non-overlapping window sequences for closed-loop DR training."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.pytorch_dataset import get_normalizer, load_split
from src.data.targets import prepare_session
from src.data.v7_dataset import build_v7_windows


class StateSequenceDataset(Dataset):
    def __init__(self, sessions, dataset, normalizer, sequence_length=5, window_size=20):
        self.records, self.index = [], []
        for session in sessions:
            imu, targets = prepare_session(dataset, session)
            record = build_v7_windows(imu.to_numpy(), targets, window_size=window_size, stride=window_size)
            if record is None or len(record["imu"]) < sequence_length:
                continue
            record["imu"] = normalizer.transform_imu(record["imu"])
            for name in ("speed", "initial_speed", "mean_speed"):
                record[name] = normalizer.transform_speed(record[name])
            record["position"] = normalizer.transform_position(record["position"])
            record["yaw_rate"] = normalizer.transform_yaw(record["yaw_rate"])
            record_id = len(self.records); self.records.append(record)
            self.index.extend((record_id, start) for start in range(len(record["imu"]) - sequence_length + 1))
        self.sequence_length = sequence_length
        if not self.index: raise RuntimeError("No V9 sequences were created.")

    def __len__(self): return len(self.index)

    def __getitem__(self, item):
        record_id, start = self.index[item]; r = self.records[record_id]; sl = slice(start, start + self.sequence_length)
        return {"imu": torch.from_numpy(r["imu"][sl].astype(np.float32)),
                "initial_speed": torch.tensor(r["initial_speed"][start], dtype=torch.float32),
                "speed": torch.from_numpy(r["speed"][sl].astype(np.float32)),
                "position": torch.from_numpy(r["position"][sl].astype(np.float32)),
                "heading_delta": torch.from_numpy(r["heading_delta"][sl].astype(np.float32)),
                "motion": torch.from_numpy(r["motion"][sl].astype(np.int64))}


def create_v9_dataloader(split_name, batch_size=32, sequence_length=5, shuffle=False):
    normalizer = get_normalizer(); dataset, sessions = load_split(split_name)
    sequence_dataset = StateSequenceDataset(sessions, dataset, normalizer, sequence_length=sequence_length)
    return DataLoader(sequence_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available()), normalizer
