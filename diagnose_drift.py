"""
Does Driver E suffer a clock-RATE mismatch rather than a constant offset?

diagnose_sync.py found a clean split by driver once a per-session lag is applied:

    Driver A  S1 0.948   S2 0.919   S3a 0.974   S3c 0.996      strong
    Driver B  M  0.778                                          strong
    Driver D  Y1 0.800                                          strong
    Driver E  Vf / Vta / Vtb / Vw   0.05 - 0.36                 weak

Driver E is 44 of the 72 sessions, so whether it is recoverable decides how much data is
trainable. A constant lag already failed there, which points at a rate difference. There is
direct evidence for one: in session M the smartphone timestamps span 6176 s while the
vehicle spans 10597 s, yet both files were truncated to exactly the same row count. Row
counts match exactly in every session, so the streams were forced to equal length rather
than genuinely synchronised.

This fits a linear time warp, phone_index = scale * vehicle_index + offset, by scanning
scale and solving offset by cross-correlation at each step.

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

# Largest Driver E sessions, plus one known-good control from Driver A.
TARGETS = ["Vw4", "Vfa02", "Vtb5", "Vw2", "Vta29", "S3c"]
SCALES = np.concatenate([
    np.arange(0.50, 0.99, 0.02),
    np.arange(0.99, 1.02, 0.005),
    np.arange(1.02, 2.05, 0.02),
])


def col(frame, name):
    return pd.to_numeric(frame[resolve(frame, name)], errors="coerce").to_numpy(dtype=np.float64)


def standardise(values):
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    centred = values - np.nanmean(values)
    return centred / (np.nanstd(centred) + 1e-12)


def best_lag_correlation(x, y, max_lag):
    """Peak |normalised cross-correlation| over integer lags, via FFT."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    size = 1
    while size < 2 * n:
        size *= 2
    raw = np.fft.irfft(np.fft.rfft(x, size) * np.conjugate(np.fft.rfft(y, size)), size)
    max_lag = min(max_lag, n // 3)
    stacked = np.concatenate([raw[-max_lag:], raw[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    overlap = n - np.abs(lags)
    keep = overlap >= 600
    if not keep.any():
        return 0.0, 0
    values = stacked[keep] / overlap[keep]
    lags = lags[keep]
    position = int(np.argmax(np.abs(values)))
    return float(values[position]), int(lags[position])


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = {s["session_id"]: s for s in dataset.get_sessions()}

    header = (f"{'session':<9} {'rows':>7} {'base|c|':>8} {'bestScale':>10} "
              f"{'bestLag':>8} {'best|c|':>8} {'gain':>7}")
    print(header)
    print("-" * len(header))

    for session_id in TARGETS:
        session = sessions.get(session_id)
        if session is None:
            continue
        smartphone, vehicle = dataset.load_session(session)
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        n = min(len(smartphone), len(vehicle))
        gyro_pitch = col(smartphone, "GYROSCOPE Pitch")[:n]
        vehicle_yaw = np.radians(col(vehicle, "Yaw Rate")[:n])
        if np.nanstd(vehicle_yaw) < 1e-3:
            continue

        reference = standardise(vehicle_yaw)
        base_corr, _ = best_lag_correlation(standardise(gyro_pitch), reference, 1200)

        phone_axis = np.arange(n, dtype=np.float64)
        best = (0.0, 1.0, 0)
        for scale in SCALES:
            # Resample the phone channel onto a vehicle-rate timebase.
            sampled_at = phone_axis * scale
            inside = sampled_at <= phone_axis[-1]
            if inside.sum() < 2000:
                continue
            warped = np.interp(sampled_at[inside], phone_axis, gyro_pitch)
            corr, lag = best_lag_correlation(standardise(warped), reference[:inside.sum()], 600)
            if abs(corr) > abs(best[0]):
                best = (corr, float(scale), lag)

        gain = abs(best[0]) - abs(base_corr)
        print(f"{session_id:<9} {n:>7} {abs(base_corr):>8.3f} {best[1]:>10.3f} "
              f"{best[2]:>8} {abs(best[0]):>8.3f} {gain:>7.3f}")

    print("\nA bestScale far from 1.0 with a large gain means a genuine rate mismatch.")
    print("A bestScale near 1.0 with little gain means the disagreement is not a linear")
    print("time warp, and those sessions cannot be rescued this way.")


if __name__ == "__main__":
    main()
