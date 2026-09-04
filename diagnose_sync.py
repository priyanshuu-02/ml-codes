"""
Can per-session synchronisation recover the whole dataset?

Established so far:
  * The smartphone IMU is real and healthy in every session: gyro std 0.03-0.40 rad/s,
    linear acceleration std 0.7-2.3 m/s^2, zero duplicated rows, zero flat rows.
  * Yet only 3 of 44 sessions show the phone gyro tracking the vehicle's yaw rate at lag 0.
  * Phone GPS SPEED is a poor alignment probe because GPS has its own latency: applying the
    lag it suggests destroyed the gyro agreement on S1, dropping it from 0.932 to 0.030.

So the streams are misaligned rather than empty, and alignment must be estimated from two
instantaneous rotation measurements: the phone gyro and the vehicle's Yaw Rate.

This scans lag per session by FFT cross-correlation. If a lag exists that lifts the weak
sessions to strong agreement, the dataset is fully usable once synchronised, and the missing
synchronisation is the real reason the original model could not learn from the IMU.

Measures only. Writes a report so the training pipeline can consume the offsets later.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset
from src.data.columns import resolve

DATASET_ROOT = "data/IO-VNBD/synchronized"
REPORT = Path("outputs/session_sync.json")

# Ten seconds either way covers plausible logging offsets without letting the search
# wander into spurious matches on long records.
MAX_LAG_SAMPLES = 1200
MIN_OVERLAP = 800


def col(frame, name):
    return pd.to_numeric(frame[resolve(frame, name)], errors="coerce").to_numpy(dtype=np.float64)


def standardise(values):
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    centred = values - np.nanmean(values)
    scale = np.nanstd(centred)
    return centred / (scale + 1e-12)


def cross_correlate(x, y, max_lag):
    """
    Normalised cross-correlation over integer lags via FFT.

    Returns (lags, correlations) where a positive lag means y must be advanced to line up
    with x, matching the convention used when slicing the arrays afterwards.
    """
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.rfft(x, size)
    fy = np.fft.rfft(y, size)
    raw = np.fft.irfft(fx * np.conjugate(fy), size)
    raw = np.concatenate([raw[-max_lag:], raw[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    # Normalise by the overlap length at each lag rather than by n, so long shifts are
    # not artificially favoured.
    overlap = n - np.abs(lags)
    valid = overlap >= MIN_OVERLAP
    correlations = np.full(len(lags), np.nan)
    correlations[valid] = raw[valid] / overlap[valid]
    return lags, correlations


def evaluate(dataset, session):
    smartphone, vehicle = dataset.load_session(session)
    smartphone.columns = smartphone.columns.astype(str).str.strip()
    vehicle.columns = vehicle.columns.astype(str).str.strip()

    n = min(len(smartphone), len(vehicle))
    if n < 2 * MIN_OVERLAP:
        return None

    gyro = np.column_stack([
        col(smartphone, "GYROSCOPE Yaw"),
        col(smartphone, "GYROSCOPE Pitch"),
        col(smartphone, "GYROSCOPE Roll"),
    ])[:n]
    vehicle_yaw = np.radians(col(vehicle, "Yaw Rate")[:n])
    speed = col(vehicle, "Velocity")[:n] / 3.6

    if not np.isfinite(vehicle_yaw).any() or np.nanstd(vehicle_yaw) < 1e-3:
        return None

    reference = standardise(vehicle_yaw)
    best = {"corr": 0.0, "lag": 0, "channel": None, "zero": 0.0}

    for index, name in enumerate(("Yaw", "Pitch", "Roll")):
        channel = standardise(gyro[:, index])
        lags, correlations = cross_correlate(channel, reference, min(MAX_LAG_SAMPLES, n // 3))
        if np.all(np.isnan(correlations)):
            continue
        magnitude = np.abs(correlations)
        position = int(np.nanargmax(magnitude))
        zero_position = int(np.where(lags == 0)[0][0])
        if magnitude[position] > abs(best["corr"]):
            best = {
                "corr": float(correlations[position]),
                "lag": int(lags[position]),
                "channel": name,
                "zero": float(correlations[zero_position]),
            }

    if best["channel"] is None:
        return None

    return {
        "session_id": session["session_id"],
        "category": session["category"],
        "rows": int(n),
        "moving_rows": int(np.sum(speed > 3.0)),
        "channel": best["channel"],
        "lag_samples": best["lag"],
        "lag_seconds": round(best["lag"] / 10.0, 2),
        "corr_at_lag": round(best["corr"], 4),
        "corr_at_zero": round(best["zero"], 4),
        "yaw_rate_std": round(float(np.nanstd(vehicle_yaw)), 4),
    }


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}\n")

    header = (f"{'session':<10} {'rows':>7} {'ch':<6} {'lag':>6} {'lag s':>7} "
              f"{'|corr|@lag':>11} {'corr@0':>8} {'gain':>7}")
    print(header)
    print("-" * len(header))

    records = []
    for session in sessions:
        try:
            result = evaluate(dataset, session)
        except Exception as error:  # noqa: BLE001 - keep sweeping
            print(f"{session['session_id']:<10} failed: {error}")
            continue
        if result is None:
            continue
        gain = abs(result["corr_at_lag"]) - abs(result["corr_at_zero"])
        print(f"{result['session_id']:<10} {result['rows']:>7} {result['channel']:<6} "
              f"{result['lag_samples']:>6} {result['lag_seconds']:>7.1f} "
              f"{abs(result['corr_at_lag']):>11.3f} {result['corr_at_zero']:>8.3f} {gain:>7.3f}")
        records.append(result)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if not records:
        print("No sessions could be evaluated.")
        return

    at_lag = np.array([abs(r["corr_at_lag"]) for r in records])
    at_zero = np.array([abs(r["corr_at_zero"]) for r in records])
    lags = np.array([r["lag_samples"] for r in records])

    strong_zero = int(np.sum(at_zero > 0.6))
    strong_lag = int(np.sum(at_lag > 0.6))
    print(f"sessions evaluated          : {len(records)}")
    print(f"strong at lag 0             : {strong_zero}")
    print(f"strong after lag correction : {strong_lag}")
    print(f"mean |corr| at lag 0        : {at_zero.mean():.3f}")
    print(f"mean |corr| after lag       : {at_lag.mean():.3f}")
    print(f"median |lag|                : {np.median(np.abs(lags)):.0f} samples "
          f"({np.median(np.abs(lags))/10.0:.1f} s)")
    print(f"lags hitting the search cap : {int(np.sum(np.abs(lags) >= MAX_LAG_SAMPLES - 1))}")

    rows_total = sum(r["rows"] for r in records)
    rows_strong = sum(r["rows"] for r in records if abs(r["corr_at_lag"]) > 0.6)
    print(f"rows total                  : {rows_total:,}")
    print(f"rows usable after sync      : {rows_strong:,} "
          f"({100*rows_strong/max(rows_total,1):.1f}%)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT}")

    print()
    if strong_lag > strong_zero * 3:
        print("VERDICT: per-session synchronisation recovers most of the dataset. The")
        print("missing synchronisation is the reason the IMU could not predict vehicle")
        print("motion, and it must be applied before training.")
    elif strong_lag > strong_zero:
        print("VERDICT: synchronisation helps but does not recover everything. Train only")
        print("on sessions that pass a correspondence check.")
    else:
        print("VERDICT: a constant lag does not explain the disagreement. A clock-rate")
        print("difference rather than a fixed offset is the likely cause.")


if __name__ == "__main__":
    main()
