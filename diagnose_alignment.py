"""
Is the smartphone stream actually row-aligned with the vehicle stream?

This is the question that decides whether the model can learn anything at all. Earlier
diagnostics validated the displacement TARGET, but that used only vehicle columns
(lat/lon/heading/velocity), so it says nothing about whether the smartphone IMU rows line
up with the vehicle rows they are paired against. If they do not, every training sample
pairs an IMU window with a target from a different moment and no architecture can help.

Both files record speed independently:
    smartphone : GPS SPEED (Kmh)
    vehicle    : Velocity (km/hr)

Speed is strong, smooth and unambiguous, which makes it the ideal alignment probe. A
correlation near 1.0 at zero lag means the rows correspond. Anything else localises the
problem precisely.

Also checks each raw gyro channel against the vehicle's Yaw Rate, since "GYROSCOPE Yaw"
may already be a resolved yaw rate rather than a phone-axis component.

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


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def series(frame, name):
    return pd.to_numeric(frame[resolve(frame, name)], errors="coerce").to_numpy(dtype=np.float64)


def best_lag(x, y, max_lag=400):
    """Normalised cross-correlation over integer lags. Positive lag means y lags x."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 500:
        return None
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)

    results = {}
    best = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = x[-lag:], y[:len(y) + lag]
        elif lag > 0:
            a, b = x[:len(x) - lag], y[lag:]
        else:
            a, b = x, y
        m = min(len(a), len(b))
        if m < 500:
            continue
        c = float(np.mean(a[:m] * b[:m]))
        if lag == 0:
            results["zero"] = c
        if c > best[0]:
            best = (c, lag)
    results["best"] = best[0]
    results["lag"] = best[1]
    return results


def speed_alignment(dataset, sessions, limit=16):
    rule("1. SPEED ALIGNMENT: smartphone GPS SPEED vs vehicle Velocity")
    print("Both files record speed. Aligned rows give a correlation near 1.0 at lag 0.\n")
    print(f"{'session':<10} {'rows':>8} {'corr@0':>8} {'corr@best':>10} {'lag':>6} {'lag s':>7}")

    zero, best, lags = [], [], []
    for session in sessions[:limit]:
        smartphone, vehicle = dataset.load_session(session)
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        n = min(len(smartphone), len(vehicle))
        phone_speed = series(smartphone, "GPS SPEED")[:n]
        vehicle_speed = series(vehicle, "Velocity")[:n]

        result = best_lag(phone_speed, vehicle_speed)
        if result is None:
            continue
        print(f"{session['session_id']:<10} {n:>8} {result['zero']:>8.3f} {result['best']:>10.3f} "
              f"{result['lag']:>6} {result['lag']/10.0:>7.1f}")
        zero.append(result["zero"])
        best.append(result["best"])
        lags.append(result["lag"])

    if zero:
        print(f"\nmean corr at lag 0     : {np.nanmean(zero):+.3f}")
        print(f"mean corr at best lag  : {np.nanmean(best):+.3f}")
        print(f"median |best lag|      : {np.median(np.abs(lags)):.0f} samples "
              f"({np.median(np.abs(lags))/10.0:.1f} s)")
        if np.nanmean(zero) > 0.9:
            print("VERDICT: streams are row-aligned.")
        elif np.nanmean(best) > 0.9:
            print("VERDICT: streams are MISALIGNED by a per-session lag, but correctable.")
        else:
            print("VERDICT: speed does not correspond even after lag search.")
    return zero, best, lags


def gyro_channel_check(dataset, sessions, limit=12):
    rule("2. WHICH GYRO CHANNEL IS THE YAW RATE?")
    print("If 'GYROSCOPE Yaw' is already a resolved yaw rate, it should track the")
    print("vehicle's Yaw Rate directly with no projection.\n")
    print(f"{'session':<10} {'Yaw':>8} {'Pitch':>8} {'Roll':>8} {'best ch':>9} {'lag':>6}")

    for session in sessions[:limit]:
        smartphone, vehicle = dataset.load_session(session)
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        n = min(len(smartphone), len(vehicle))
        reference = np.radians(series(vehicle, "Yaw Rate")[:n])
        speed = series(vehicle, "Velocity")[:n] / 3.6
        moving = speed > 2.0
        if moving.sum() < 500 or np.nanstd(reference[moving]) < 1e-6:
            continue

        scores = {}
        for channel in ("Yaw", "Pitch", "Roll"):
            values = series(smartphone, f"GYROSCOPE {channel}")[:n]
            ok = moving & np.isfinite(values) & np.isfinite(reference)
            scores[channel] = float(np.corrcoef(values[ok], reference[ok])[0, 1]) if ok.sum() > 500 else np.nan

        winner = max(scores, key=lambda k: abs(scores[k]) if np.isfinite(scores[k]) else -1)
        winner_values = series(smartphone, f"GYROSCOPE {winner}")[:n]
        lag = best_lag(np.abs(winner_values), np.abs(reference))
        lag_value = lag["lag"] if lag else 0

        print(f"{session['session_id']:<10} {scores['Yaw']:>8.3f} {scores['Pitch']:>8.3f} "
              f"{scores['Roll']:>8.3f} {winner:>9} {lag_value:>6}")


def within_file_consistency(dataset, sessions, limit=10):
    """
    Sanity-check each file against itself, which isolates whether a stream is internally
    coherent regardless of cross-file alignment.
    """
    rule("3. WITHIN-FILE CONSISTENCY")
    print("smartphone: GPS SPEED vs distance derived from its own GPS positions")
    print("vehicle   : Velocity vs distance derived from its own positions\n")
    print(f"{'session':<10} {'phone corr':>11} {'vehicle corr':>13}")

    earth = 6371000.0
    for session in sessions[:limit]:
        smartphone, vehicle = dataset.load_session(session)
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        def derived_speed(frame, lat_name, lon_name, hz=10.0):
            lat = series(frame, lat_name)
            lon = series(frame, lon_name)
            lat0 = np.nanmean(lat)
            east = np.radians(lon) * earth * np.cos(np.radians(lat0))
            north = np.radians(lat) * earth
            step = 10
            d = np.hypot(east[step:] - east[:-step], north[step:] - north[:-step])
            return d / (step / hz)

        phone_derived = derived_speed(smartphone, "GPS LATITUDE", "GPS LONGITUDE")
        phone_reported = (series(smartphone, "GPS SPEED") / 3.6)[:len(phone_derived)]
        ok = np.isfinite(phone_derived) & np.isfinite(phone_reported) & (phone_reported > 2.0)
        phone_corr = float(np.corrcoef(phone_derived[ok], phone_reported[ok])[0, 1]) if ok.sum() > 500 else np.nan

        vehicle_derived = derived_speed(vehicle, "Latitude", "Longitude")
        vehicle_reported = (series(vehicle, "Velocity") / 3.6)[:len(vehicle_derived)]
        ok2 = np.isfinite(vehicle_derived) & np.isfinite(vehicle_reported) & (vehicle_reported > 2.0)
        vehicle_corr = float(np.corrcoef(vehicle_derived[ok2], vehicle_reported[ok2])[0, 1]) if ok2.sum() > 500 else np.nan

        print(f"{session['session_id']:<10} {phone_corr:>11.3f} {vehicle_corr:>13.3f}")


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}")
    speed_alignment(dataset, sessions)
    gyro_channel_check(dataset, sessions)
    within_file_consistency(dataset, sessions)


if __name__ == "__main__":
    main()
