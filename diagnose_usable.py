"""
How much of the dataset has smartphone IMU genuinely corresponding to vehicle ground truth?

Two earlier results looked contradictory:

  * Phone GPS SPEED versus vehicle Velocity peaked at lags of tens of samples, which
    suggested the streams were misaligned.
  * Applying that lag COLLAPSED the gyro-to-yaw-rate correlation, from 0.932 to 0.030 on
    session S1.

The resolution is that the rows are aligned and the phone's GPS SPEED column is simply
delayed by GPS processing latency. Speed is therefore a bad alignment probe, while the
gyro-to-yaw-rate relationship at lag 0 is a good one: it is instantaneous, needs no frame,
and is strong during turns.

This sweeps every session at lag 0 to establish how much data is actually trainable, and
also whether the gyro triple behaves as a vector or as Euler angle rates. Euler rates
cannot be rotated or projected like a vector, which would explain why projecting onto
gravity-derived Down fails while a single channel succeeds.

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


def series(frame, name):
    return pd.to_numeric(frame[resolve(frame, name)], errors="coerce").to_numpy(dtype=np.float64)


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}\n")

    print(f"{'session':<10} {'rows':>8} {'moving':>8} {'turnStd':>8} "
          f"{'gYaw':>7} {'gPitch':>7} {'gRoll':>7} {'proj':>7} {'best':>7}")

    records = []
    for session in sessions:
        try:
            smartphone, vehicle = dataset.load_session(session)
        except Exception as error:  # noqa: BLE001 - report and continue the sweep
            print(f"{session['session_id']:<10} load failed: {error}")
            continue
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        n = min(len(smartphone), len(vehicle))
        if n < 1000:
            continue

        gyro = np.column_stack([
            series(smartphone, "GYROSCOPE Yaw"),
            series(smartphone, "GYROSCOPE Pitch"),
            series(smartphone, "GYROSCOPE Roll"),
        ])[:n]
        gravity = np.column_stack([series(smartphone, f"GRAVITY {ax}") for ax in "XYZ"])[:n]
        yaw_rate = np.radians(series(vehicle, "Yaw Rate")[:n])
        speed = series(vehicle, "Velocity")[:n] / 3.6

        ok = (np.isfinite(gyro).all(axis=1) & np.isfinite(gravity).all(axis=1)
              & np.isfinite(yaw_rate) & np.isfinite(speed) & (speed > 3.0))
        moving = int(ok.sum())
        turn_std = float(np.std(yaw_rate[ok])) if moving > 100 else 0.0
        if moving < 500 or turn_std < 1e-3:
            print(f"{session['session_id']:<10} {n:>8} {moving:>8} {turn_std:>8.4f} "
                  f"{'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7}   (too little turning)")
            records.append((session["session_id"], n, moving, turn_std, np.nan, np.nan))
            continue

        channel = []
        for index in range(3):
            c = float(np.corrcoef(gyro[ok][:, index], yaw_rate[ok])[0, 1])
            channel.append(c if np.isfinite(c) else 0.0)

        down = -gravity[ok] / np.maximum(np.linalg.norm(gravity[ok], axis=1, keepdims=True), 1e-9)
        projected = np.einsum("ni,ni->n", gyro[ok], down)
        proj_corr = float(np.corrcoef(projected, yaw_rate[ok])[0, 1]) if projected.std() > 1e-12 else 0.0
        best = max(np.abs(channel))

        print(f"{session['session_id']:<10} {n:>8} {moving:>8} {turn_std:>8.4f} "
              f"{channel[0]:>7.3f} {channel[1]:>7.3f} {channel[2]:>7.3f} "
              f"{proj_corr:>7.3f} {best:>7.3f}")
        records.append((session["session_id"], n, moving, turn_std, best, abs(proj_corr)))

    usable = [r for r in records if np.isfinite(r[4])]
    strong = [r for r in usable if r[4] > 0.6]
    weak = [r for r in usable if r[4] <= 0.6]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"sessions evaluated            : {len(records)}")
    print(f"with enough turning to judge  : {len(usable)}")
    print(f"strong IMU/vehicle agreement  : {len(strong)}  (best channel |corr| > 0.6)")
    print(f"weak agreement                : {len(weak)}")
    if usable:
        best_values = np.array([r[4] for r in usable])
        proj_values = np.array([r[5] for r in usable])
        print(f"mean best-channel |corr|      : {np.nanmean(best_values):.3f}")
        print(f"mean projected |corr|         : {np.nanmean(proj_values):.3f}")
        print()
        if np.nanmean(best_values) > np.nanmean(proj_values) * 2:
            print("The single-channel relationship is far stronger than the projected one.")
            print("That is the signature of EULER ANGLE RATES rather than body angular")
            print("velocity: Euler rates are not a vector, so projecting them onto Down is")
            print("not meaningful. Gravity-based rotation of the gyro triple is therefore")
            print("unsafe for this dataset.")
    total_rows = sum(r[1] for r in records)
    strong_rows = sum(r[1] for r in strong)
    print(f"\nrows total                    : {total_rows:,}")
    print(f"rows in strong sessions       : {strong_rows:,} ({100*strong_rows/max(total_rows,1):.1f}%)")


if __name__ == "__main__":
    main()
