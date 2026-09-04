"""
Establish the relationship between the gyro columns and the gravity columns.

diagnose_alignment.py produced a contradiction that has to be resolved before any
training data is built:

  * Projecting the gyro vector onto Down derived from the GRAVITY column correlates
    ~0.02 with the vehicle's Yaw Rate, which is nothing.
  * The single column labelled "GYROSCOPE Pitch" correlates up to 0.95 with it.

Projection onto Down is the correct way to extract yaw rate and is invariant to how the
axes are named, so it can only fail if the gyro triple and the gravity triple are not
expressed in the same axis order. This script finds the permutation and signs that
reconcile them, which is what makes a real vehicle frame possible.

Sessions are lag-corrected first, since the same diagnostic showed per-session offsets of
up to 11 s.

Measures only.
"""
import itertools
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


def estimate_lag(x, y, max_lag=200):
    """Lag that best aligns y to x, using speed which is smooth and unambiguous."""
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    best, best_lag = -2.0, 0
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
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


def load_aligned(dataset, session):
    """Load a session with the smartphone stream shifted onto the vehicle stream."""
    smartphone, vehicle = dataset.load_session(session)
    smartphone.columns = smartphone.columns.astype(str).str.strip()
    vehicle.columns = vehicle.columns.astype(str).str.strip()

    n = min(len(smartphone), len(vehicle))
    phone_speed = series(smartphone, "GPS SPEED")[:n]
    vehicle_speed = series(vehicle, "Velocity")[:n]
    lag, quality = estimate_lag(phone_speed, vehicle_speed)

    gyro = np.column_stack([
        series(smartphone, "GYROSCOPE Yaw"),
        series(smartphone, "GYROSCOPE Pitch"),
        series(smartphone, "GYROSCOPE Roll"),
    ])[:n]
    gravity = np.column_stack([series(smartphone, f"GRAVITY {ax}") for ax in "XYZ"])[:n]
    accel = np.column_stack([series(smartphone, f"ACCELEROMETER {ax}") for ax in "XYZ"])[:n]
    yaw_rate = np.radians(series(vehicle, "Yaw Rate")[:n])
    speed = vehicle_speed / 3.6

    # Shift the phone arrays so index i of both streams refers to the same instant.
    if lag > 0:
        gyro, gravity, accel = gyro[:n - lag], gravity[:n - lag], accel[:n - lag]
        yaw_rate, speed = yaw_rate[lag:], speed[lag:]
    elif lag < 0:
        gyro, gravity, accel = gyro[-lag:], gravity[-lag:], accel[-lag:]
        yaw_rate, speed = yaw_rate[:n + lag], speed[:n + lag]

    m = min(len(gyro), len(yaw_rate))
    return {
        "gyro": gyro[:m], "gravity": gravity[:m], "accel": accel[:m],
        "yaw_rate": yaw_rate[:m], "speed": speed[:m],
        "lag": lag, "lag_quality": quality,
    }


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def permutation_search(dataset, sessions, limit=14):
    """
    Which reordering and sign flip of the gyro triple, projected on Down, reproduces the
    vehicle's yaw rate?
    """
    rule("1. GYRO AXIS PERMUTATION SEARCH")
    print("Projecting the reordered gyro onto Down from GRAVITY and correlating with the")
    print("vehicle Yaw Rate. Projection is name-invariant, so a clear winner identifies")
    print("the true axis order.\n")

    permutations = list(itertools.permutations(range(3)))
    signs = list(itertools.product([1, -1], repeat=3))
    scores = {}
    usable = 0

    for session in sessions[:limit]:
        d = load_aligned(dataset, session)
        ok = (np.isfinite(d["gyro"]).all(axis=1) & np.isfinite(d["gravity"]).all(axis=1)
              & np.isfinite(d["yaw_rate"]) & (d["speed"] > 3.0))
        if ok.sum() < 800 or np.nanstd(d["yaw_rate"][ok]) < 1e-4:
            continue
        if d["lag_quality"] < 0.75:
            continue
        usable += 1

        down = -unit(d["gravity"][ok])
        reference = d["yaw_rate"][ok]

        for perm in permutations:
            for sign in signs:
                candidate = d["gyro"][ok][:, list(perm)] * np.array(sign)
                projected = np.einsum("ni,ni->n", candidate, down)
                if projected.std() < 1e-9:
                    continue
                c = float(np.corrcoef(projected, reference)[0, 1])
                if np.isfinite(c):
                    scores.setdefault((perm, sign), []).append(c)

    if not scores:
        print("No sessions met the alignment-quality bar.")
        return None

    ranked = sorted(
        ((np.mean(v), np.min(v), len(v), k) for k, v in scores.items() if len(v) >= max(3, usable // 2)),
        key=lambda t: -t[0],
    )
    print(f"sessions used: {usable}\n")
    print(f"{'perm':<12} {'signs':<14} {'mean corr':>10} {'min corr':>9} {'n':>4}")
    for mean_c, min_c, n, (perm, sign) in ranked[:8]:
        print(f"{str(perm):<12} {str(sign):<14} {mean_c:>10.3f} {min_c:>9.3f} {n:>4}")

    if ranked:
        best = ranked[0]
        print(f"\nBEST: permutation {best[3][0]} signs {best[3][1]} "
              f"mean corr {best[0]:.3f} min {best[1]:.3f}")
        if best[0] > 0.8:
            print("VERDICT: gyro axis order recovered; a true vehicle frame is buildable.")
        elif best[0] > 0.5:
            print("VERDICT: partial agreement only. Gyro and gravity frames are related but")
            print("         not by a clean permutation, so gravity-only Down is unsafe.")
        else:
            print("VERDICT: no permutation reconciles the two triples.")
    return ranked


def single_channel_baseline(dataset, sessions, limit=14):
    """For comparison: the best single gyro column, lag corrected."""
    rule("2. SINGLE-CHANNEL BASELINE, LAG CORRECTED")
    print(f"{'session':<10} {'lag':>5} {'quality':>8} {'Yaw':>8} {'Pitch':>8} {'Roll':>8}")
    per_channel = {0: [], 1: [], 2: []}
    for session in sessions[:limit]:
        d = load_aligned(dataset, session)
        ok = (np.isfinite(d["gyro"]).all(axis=1) & np.isfinite(d["yaw_rate"]) & (d["speed"] > 3.0))
        if ok.sum() < 800 or np.nanstd(d["yaw_rate"][ok]) < 1e-4:
            continue
        row = []
        for index in range(3):
            c = float(np.corrcoef(d["gyro"][ok][:, index], d["yaw_rate"][ok])[0, 1])
            row.append(c)
            if d["lag_quality"] >= 0.75:
                per_channel[index].append(c)
        print(f"{session['session_id']:<10} {d['lag']:>5} {d['lag_quality']:>8.3f} "
              f"{row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f}")

    print()
    for index, name in enumerate(("Yaw", "Pitch", "Roll")):
        values = per_channel[index]
        if values:
            print(f"  {name:<6} mean {np.mean(values):+.3f}  mean|.| {np.mean(np.abs(values)):.3f}  n={len(values)}")


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}")
    single_channel_baseline(dataset, sessions)
    permutation_search(dataset, sessions)


if __name__ == "__main__":
    main()
