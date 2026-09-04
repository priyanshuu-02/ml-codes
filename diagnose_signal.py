"""
Is the smartphone IMU signal actually present in the sessions that showed weak agreement?

diagnose_usable.py found that only 3 of 44 judgeable sessions have smartphone IMU that
tracks the vehicle's own yaw rate. Two very different causes would produce that:

  A. The IMU data is real but not time-aligned with the vehicle log in those sessions.
  B. The IMU data is flat, duplicated or dead, so there is nothing to align.

The two are distinguishable. Real driving produces gyro standard deviation on the order of
0.1 rad/s and linear acceleration spread of order 1 m/s^2. Duplicated consecutive rows
indicate a stream that was padded or upsampled to match the vehicle row count, which would
also explain why S and V row counts match exactly in every session.

Measures only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset
from src.data.columns import resolve

DATASET_ROOT = "data/IO-VNBD/synchronized"

# Mix of the three sessions that agreed and several that did not.
STRONG = ["S1", "S3c", "M"]
WEAK = ["S2", "Vfa02", "Vw2", "Vw4", "Vtb5", "Y1", "Vta29", "S4"]


def col(frame, name):
    return pd.to_numeric(frame[resolve(frame, name)], errors="coerce").to_numpy(dtype=np.float64)


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = {s["session_id"]: s for s in dataset.get_sessions()}

    header = (f"{'session':<9} {'grp':<6} {'gyroStdY':>9} {'gyroStdP':>9} {'gyroStdR':>9} "
              f"{'linAccStd':>10} {'vYawStd':>8} {'dupRow%':>8} {'flat%':>7}")
    print(header)
    print("-" * len(header))

    for group, ids in (("strong", STRONG), ("weak", WEAK)):
        for session_id in ids:
            session = sessions.get(session_id)
            if session is None:
                continue
            smartphone, vehicle = dataset.load_session(session)
            smartphone.columns = smartphone.columns.astype(str).str.strip()
            vehicle.columns = vehicle.columns.astype(str).str.strip()

            n = min(len(smartphone), len(vehicle))
            gyro = np.column_stack([
                col(smartphone, "GYROSCOPE Yaw"),
                col(smartphone, "GYROSCOPE Pitch"),
                col(smartphone, "GYROSCOPE Roll"),
            ])[:n]
            accel = np.column_stack([col(smartphone, f"ACCELEROMETER {ax}") for ax in "XYZ"])[:n]
            gravity = np.column_stack([col(smartphone, f"GRAVITY {ax}") for ax in "XYZ"])[:n]
            linear = accel - gravity
            vehicle_yaw = np.radians(col(vehicle, "Yaw Rate")[:n])

            finite = np.isfinite(gyro).all(axis=1)
            duplicated = np.zeros(n, dtype=bool)
            duplicated[1:] = np.all(np.isclose(gyro[1:], gyro[:-1], equal_nan=True), axis=1)
            flat = np.all(np.abs(gyro) < 1e-9, axis=1)

            std = np.nanstd(gyro, axis=0)
            print(f"{session_id:<9} {group:<6} {std[0]:>9.4f} {std[1]:>9.4f} {std[2]:>9.4f} "
                  f"{np.nanstd(np.linalg.norm(linear, axis=1)):>10.4f} "
                  f"{np.nanstd(vehicle_yaw):>8.4f} "
                  f"{100 * duplicated.mean():>8.1f} {100 * flat.mean():>7.1f}")

    print("\nReal driving: gyro std ~0.05-0.3 rad/s, linear accel std ~0.5-2 m/s^2.")
    print("A high duplicated-row percentage means the stream was padded or upsampled,")
    print("which destroys the sample-to-sample correspondence the model needs.")


if __name__ == "__main__":
    main()
