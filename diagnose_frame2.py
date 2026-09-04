"""
Derive the phone-to-vehicle rotation from physics instead of from the ORIENTATION columns.

diagnose_frame.py showed that reconstructing Android's rotation matrix from the dataset's
ORIENTATION (Yaw/Pitch/Roll) columns leaves gravity roughly 96 degrees away from vertical,
so those columns do not follow the convention Android's getOrientation() produces. Rather
than guessing among Euler orders and sign flips, the frame is built from quantities whose
meaning can be verified directly:

  DOWN comes from the GRAVITY column, which is a measured vector in phone axes. That fixes
  roll and pitch exactly and leaves exactly one unknown, the yaw about Down.

  The remaining yaw is solved in closed form against the vehicle's own longitudinal
  acceleration channel, which is the same information the runtime alignment calibrator
  learns from GNSS course.

Two independent validations, neither of which needs the yaw estimate:

  1. Gravity magnitude and stability, confirming the GRAVITY column is a real gravity vector.
  2. Angular rate projected on Down versus the vehicle's reported Yaw Rate. If Down is
     correct this correlates strongly, and its sign reveals the convention.

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
GRAVITY = 9.80665


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load(dataset, session):
    smartphone, vehicle = dataset.load_session(session)
    smartphone.columns = smartphone.columns.astype(str).str.strip()
    vehicle.columns = vehicle.columns.astype(str).str.strip()

    def s(col):
        return pd.to_numeric(smartphone[resolve(smartphone, col)], errors="coerce").to_numpy(dtype=np.float64)

    def v(col):
        return pd.to_numeric(vehicle[resolve(vehicle, col)], errors="coerce").to_numpy(dtype=np.float64)

    n = min(len(smartphone), len(vehicle))
    return {
        "acc": np.column_stack([s(f"ACCELEROMETER {ax}") for ax in "XYZ"])[:n],
        "grav": np.column_stack([s(f"GRAVITY {ax}") for ax in "XYZ"])[:n],
        "gyro": np.column_stack([s("GYROSCOPE Yaw"), s("GYROSCOPE Pitch"), s("GYROSCOPE Roll")])[:n],
        "speed": (v("Velocity") / 3.6)[:n],
        "yaw_rate": np.radians(v("Yaw Rate"))[:n],
        "long_g": v("Indicated Longitudinal Acceleration")[:n],
        "lat_g": v("Indicated Lateral Acceleration")[:n],
    }


def unit(vectors):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-9)


def horizontal_basis(up):
    """
    Two orthonormal vectors spanning the plane perpendicular to `up`, per sample.

    The seed axis is chosen to be the least aligned with `up` so the cross product never
    degenerates.
    """
    seed = np.zeros_like(up)
    least = np.argmin(np.abs(up), axis=1)
    seed[np.arange(len(up)), least] = 1.0
    e1 = unit(np.cross(seed, up))
    e2 = np.cross(up, e1)
    return e1, e2


def gravity_sanity(dataset, sessions, limit=10):
    rule("1. IS THE GRAVITY COLUMN A REAL GRAVITY VECTOR?")
    print(f"{'session':<10} {'|g| mean':>9} {'|g| std':>8} {'valid%':>7}")
    for session in sessions[:limit]:
        d = load(dataset, session)
        ok = np.isfinite(d["grav"]).all(axis=1)
        if ok.sum() < 200:
            continue
        norm = np.linalg.norm(d["grav"][ok], axis=1)
        print(f"{session['session_id']:<10} {norm.mean():>9.4f} {norm.std():>8.4f} {100*ok.mean():>7.1f}")
    print("\nA mean near 9.81 with a small spread confirms a genuine gravity vector.")


def down_axis_check(dataset, sessions, limit=12):
    """
    Validate the Down axis without needing yaw: angular rate about Down must match the
    vehicle's reported yaw rate.
    """
    rule("2. DOWN AXIS: gyro projected on Down vs the vehicle's Yaw Rate")
    print("Needs no yaw estimate. High |corr| confirms Down; the sign gives the convention.\n")
    print(f"{'session':<10} {'corr':>8} {'slope':>8} {'n':>8}")
    correlations, slopes = [], []
    for session in sessions[:limit]:
        d = load(dataset, session)
        ok = (np.isfinite(d["grav"]).all(axis=1) & np.isfinite(d["gyro"]).all(axis=1)
              & np.isfinite(d["yaw_rate"]) & (d["speed"] > 2.0))
        if ok.sum() < 500:
            continue
        up = unit(d["grav"][ok])
        rate_about_down = -np.einsum("ni,ni->n", d["gyro"][ok], up)
        reference = d["yaw_rate"][ok]
        if reference.std() < 1e-6:
            continue
        corr = np.corrcoef(rate_about_down, reference)[0, 1]
        slope = np.polyfit(reference, rate_about_down, 1)[0]
        print(f"{session['session_id']:<10} {corr:>8.3f} {slope:>8.3f} {ok.sum():>8}")
        correlations.append(corr)
        slopes.append(slope)

    if correlations:
        print(f"\nmean corr  : {np.nanmean(correlations):+.3f}")
        print(f"mean slope : {np.nanmean(slopes):+.3f}")
        if abs(np.nanmean(correlations)) > 0.7:
            sign = "positive" if np.nanmean(correlations) > 0 else "NEGATIVE"
            print(f"VERDICT: Down axis confirmed. Yaw-rate sign is {sign}.")
        else:
            print("VERDICT: Down axis not confirmed.")
    return correlations, slopes


def yaw_from_longitudinal(dataset, sessions, limit=12):
    """
    Solve the single remaining unknown, yaw about Down, in closed form.

    With horizontal basis (e1, e2), forward is (cos phi, sin phi). Forward acceleration is
    then a1 cos phi + a2 sin phi. Maximising its inner product with the vehicle's reported
    longitudinal acceleration gives phi = atan2(sum(ref a2), sum(ref a1)).
    """
    rule("3. YAW ABOUT DOWN, SOLVED IN CLOSED FORM")
    print("Fitted against the vehicle's own longitudinal acceleration channel.\n")
    print(f"{'session':<10} {'yaw deg':>8} {'corr fwd':>9} {'corr lat':>9} {'std f':>7} {'std r':>7} {'ratio':>7}")

    rows = []
    for session in sessions[:limit]:
        d = load(dataset, session)
        linear = d["acc"] - d["grav"]
        ok = (np.isfinite(linear).all(axis=1) & np.isfinite(d["grav"]).all(axis=1)
              & np.isfinite(d["long_g"]) & np.isfinite(d["lat_g"]) & (d["speed"] > 2.0))
        if ok.sum() < 500:
            continue

        up = unit(d["grav"][ok])
        e1, e2 = horizontal_basis(up)
        a1 = np.einsum("ni,ni->n", linear[ok], e1)
        a2 = np.einsum("ni,ni->n", linear[ok], e2)
        reference = d["long_g"][ok] * GRAVITY
        lateral_reference = d["lat_g"][ok] * GRAVITY

        phi = np.arctan2(np.sum(reference * a2), np.sum(reference * a1))
        forward = a1 * np.cos(phi) + a2 * np.sin(phi)
        right = -a1 * np.sin(phi) + a2 * np.cos(phi)

        corr_forward = np.corrcoef(forward, reference)[0, 1]
        corr_lateral = np.corrcoef(right, lateral_reference)[0, 1]
        ratio = right.std() / max(forward.std(), 1e-9)
        print(f"{session['session_id']:<10} {np.degrees(phi):>8.1f} {corr_forward:>9.3f} "
              f"{corr_lateral:>9.3f} {forward.std():>7.3f} {right.std():>7.3f} {ratio:>7.3f}")
        rows.append((corr_forward, corr_lateral, ratio))

    if rows:
        cf = np.array([r[0] for r in rows])
        cl = np.array([r[1] for r in rows])
        print(f"\nmean corr forward : {np.nanmean(cf):+.3f}")
        print(f"mean corr lateral : {np.nanmean(cl):+.3f}")
        if np.nanmean(cf) > 0.5:
            print("VERDICT: a physically meaningful vehicle frame is recoverable per session.")
        else:
            print("VERDICT: forward acceleration does not track the vehicle channel; the")
            print("         phone IMU and the vehicle log may not be time aligned.")
    return rows


def time_alignment_probe(dataset, sessions, limit=6, max_lag=60):
    """
    If the frame fits poorly, the usual cause is a time offset between the phone and
    vehicle logs. Scan lags and report the best.
    """
    rule("4. TIME ALIGNMENT PROBE: best lag between phone IMU and vehicle channel")
    print("Uses |linear acceleration| against |vehicle longitudinal acceleration|,")
    print("which needs no frame at all. A large best lag means the streams are offset.\n")
    print(f"{'session':<10} {'lag@best':>9} {'corr@0':>8} {'corr@best':>10}")
    for session in sessions[:limit]:
        d = load(dataset, session)
        linear = d["acc"] - d["grav"]
        magnitude = np.linalg.norm(linear, axis=1)
        reference = np.abs(d["long_g"]) * GRAVITY
        ok = np.isfinite(magnitude) & np.isfinite(reference) & (d["speed"] > 2.0)
        if ok.sum() < 2000:
            continue
        x = pd.Series(magnitude).where(ok).interpolate().to_numpy()
        y = pd.Series(reference).where(ok).interpolate().to_numpy()
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
        y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-9)

        best_lag, best = 0, -2.0
        base = np.nan
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a, b = x[-lag:], y[:len(y) + lag]
            elif lag > 0:
                a, b = x[:len(x) - lag], y[lag:]
            else:
                a, b = x, y
            m = min(len(a), len(b))
            if m < 1000:
                continue
            c = float(np.nanmean(a[:m] * b[:m]))
            if lag == 0:
                base = c
            if c > best:
                best, best_lag = c, lag
        print(f"{session['session_id']:<10} {best_lag:>9} {base:>8.3f} {best:>10.3f}")


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}")
    gravity_sanity(dataset, sessions)
    down_axis_check(dataset, sessions)
    yaw_from_longitudinal(dataset, sessions)
    time_alignment_probe(dataset, sessions)


if __name__ == "__main__":
    main()
