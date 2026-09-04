"""
Validate the phone-to-vehicle rotation before it is used to build training data.

Stage 1 of the Android work rotates IMU vectors into the vehicle FRD frame with

    v_vehicle = M(vehicleHeading) . R_enu_from_phone . v_phone

To train a model on that same representation, the dataset needs the same transform. The
dataset provides the phone's ORIENTATION (Yaw/Pitch/Roll) columns, from which Android's
device-to-ENU rotation matrix can be reconstructed, plus a GRAVITY column and the
vehicle's own longitudinal/lateral acceleration channels.

That gives two independent ways to check the transform before trusting it:

  1. GRAVITY CHECK. Rotating the measured gravity vector into ENU must produce
     (0, 0, +g): straight up. This validates roll and pitch.

  2. LONGITUDINAL ACCELERATION CHECK. The vehicle reports its own longitudinal and
     lateral acceleration. Vehicle-frame forward acceleration derived from the phone
     must correlate with the vehicle's channel. This validates yaw, which gravity
     cannot constrain.

Measures only. Trains nothing, writes no artifacts.
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


def rotation_enu_from_phone(azimuth_deg, pitch_deg, roll_deg):
    """
    Rebuild Android's device-to-ENU rotation matrix from its own Euler outputs.

    Android's getOrientation() derives the angles from the matrix as
        azimuth = atan2(R01, R11),  pitch = asin(-R21),  roll = atan2(-R20, R22)
    so this composition is its exact inverse. Verified below against gravity rather
    than taken on trust.

    Returns an array of shape (n, 3, 3) mapping phone vectors to ENU.
    """
    a = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    p = np.radians(np.asarray(pitch_deg, dtype=np.float64))
    r = np.radians(np.asarray(roll_deg, dtype=np.float64))

    sa, ca = np.sin(a), np.cos(a)
    sp, cp = np.sin(p), np.cos(p)
    sr, cr = np.sin(r), np.cos(r)

    n = a.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = ca * cr - sa * sp * sr
    R[:, 0, 1] = sa * cp
    R[:, 0, 2] = ca * sr + sa * sp * cr
    R[:, 1, 0] = -sa * cr - ca * sp * sr
    R[:, 1, 1] = ca * cp
    R[:, 1, 2] = -sa * sr + ca * sp * cr
    R[:, 2, 0] = -cp * sr
    R[:, 2, 1] = -sp
    R[:, 2, 2] = cp * cr
    return R


def enu_to_vehicle(vectors_enu, heading_deg):
    """
    ENU to vehicle FRD: X forward, Y right, Z down.

    Matches VehicleFrameTransform.enuToVehicle on the Android side and
    global_to_vehicle_frame in targets.py.
    """
    h = np.radians(np.asarray(heading_deg, dtype=np.float64))
    east, north, up = vectors_enu[:, 0], vectors_enu[:, 1], vectors_enu[:, 2]
    forward = east * np.sin(h) + north * np.cos(h)
    right = east * np.cos(h) - north * np.sin(h)
    return np.column_stack([forward, right, -up])


def load(dataset, session):
    smartphone, vehicle = dataset.load_session(session)
    smartphone.columns = smartphone.columns.astype(str).str.strip()
    vehicle.columns = vehicle.columns.astype(str).str.strip()

    def s(col):
        return pd.to_numeric(smartphone[resolve(smartphone, col)], errors="coerce").to_numpy(dtype=np.float64)

    def v(col):
        return pd.to_numeric(vehicle[resolve(vehicle, col)], errors="coerce").to_numpy(dtype=np.float64)

    n = min(len(smartphone), len(vehicle))
    data = {
        "acc": np.column_stack([s(f"ACCELEROMETER {ax}") for ax in "XYZ"])[:n],
        "grav": np.column_stack([s(f"GRAVITY {ax}") for ax in "XYZ"])[:n],
        "gyro": np.column_stack([s("GYROSCOPE Yaw"), s("GYROSCOPE Pitch"), s("GYROSCOPE Roll")])[:n],
        "azimuth": s("ORIENTATION (Yaw)")[:n],
        "pitch": s("ORIENTATION (Pitch)")[:n],
        "roll": s("ORIENTATION (Roll)")[:n],
        "heading": v("Heading")[:n],
        "speed": (v("Velocity") / 3.6)[:n],
        "yaw_rate": np.radians(v("Yaw Rate"))[:n],
        "long_g": v("Indicated Longitudinal Acceleration")[:n],
        "lat_g": v("Indicated Lateral Acceleration")[:n],
    }
    return data


def gravity_check(dataset, sessions, limit=10):
    rule("1. GRAVITY CHECK: does the reconstructed rotation level the phone?")
    print("Rotating measured gravity into ENU must give (0, 0, +9.81).\n")
    print(f"{'session':<10} {'east':>8} {'north':>8} {'up':>8} {'tilt err':>9} {'valid%':>7}")
    tilts = []
    for session in sessions[:limit]:
        d = load(dataset, session)
        ok = np.isfinite(d["grav"]).all(axis=1) & np.isfinite(d["azimuth"]) & \
             np.isfinite(d["pitch"]) & np.isfinite(d["roll"])
        if ok.sum() < 200:
            continue
        R = rotation_enu_from_phone(d["azimuth"][ok], d["pitch"][ok], d["roll"][ok])
        g_enu = np.einsum("nij,nj->ni", R, d["grav"][ok])
        norm = np.linalg.norm(g_enu, axis=1)
        # Angle between rotated gravity and straight up.
        tilt = np.degrees(np.arccos(np.clip(g_enu[:, 2] / np.maximum(norm, 1e-6), -1, 1)))
        print(f"{session['session_id']:<10} {g_enu[:,0].mean():>8.3f} {g_enu[:,1].mean():>8.3f} "
              f"{g_enu[:,2].mean():>8.3f} {np.median(tilt):>9.2f} {100*ok.mean():>7.1f}")
        tilts.append(np.median(tilt))
    if tilts:
        print(f"\nmedian tilt error across sessions: {np.median(tilts):.2f} deg")
        print("VERDICT:", "reconstruction is correct" if np.median(tilts) < 5
              else "reconstruction or orientation columns are unreliable")
    return tilts


def yaw_alignment_check(dataset, sessions, limit=10):
    """
    Validate yaw by comparing phone-derived forward acceleration with the vehicle's
    own longitudinal acceleration channel, and estimate the residual yaw offset.
    """
    rule("2. YAW CHECK: phone-derived forward accel vs the vehicle's own channel")
    print("Gravity cannot constrain yaw. The vehicle reports longitudinal acceleration,")
    print("so correlating against it validates yaw and exposes any residual offset.\n")
    print(f"{'session':<10} {'corr fwd':>9} {'best yaw':>9} {'corr@best':>10} {'n':>8}")

    results = []
    for session in sessions[:limit]:
        d = load(dataset, session)
        linear = d["acc"] - d["grav"]
        ok = (np.isfinite(linear).all(axis=1) & np.isfinite(d["azimuth"]) &
              np.isfinite(d["pitch"]) & np.isfinite(d["roll"]) &
              np.isfinite(d["heading"]) & np.isfinite(d["long_g"]) & (d["speed"] > 2.0))
        if ok.sum() < 500:
            continue

        R = rotation_enu_from_phone(d["azimuth"][ok], d["pitch"][ok], d["roll"][ok])
        linear_enu = np.einsum("nij,nj->ni", R, linear[ok])
        reference = d["long_g"][ok] * GRAVITY

        base = enu_to_vehicle(linear_enu, d["heading"][ok])
        base_corr = np.corrcoef(base[:, 0], reference)[0, 1]

        # Scan a residual yaw offset; magnetometer declination or a rotated mount
        # shows up here as a constant bias.
        best_offset, best_corr = 0.0, -2.0
        for offset in range(-180, 180, 5):
            candidate = enu_to_vehicle(linear_enu, d["heading"][ok] + offset)
            c = np.corrcoef(candidate[:, 0], reference)[0, 1]
            if np.isfinite(c) and c > best_corr:
                best_corr, best_offset = c, float(offset)

        print(f"{session['session_id']:<10} {base_corr:>9.3f} {best_offset:>9.0f} "
              f"{best_corr:>10.3f} {ok.sum():>8}")
        results.append((base_corr, best_offset, best_corr))

    if results:
        base = np.array([r[0] for r in results])
        offsets = np.array([r[1] for r in results])
        best = np.array([r[2] for r in results])
        print(f"\nmean corr at zero offset : {np.nanmean(base):.3f}")
        print(f"mean corr at best offset : {np.nanmean(best):.3f}")
        print(f"best offsets             : {offsets.tolist()}")
        if np.nanmean(base) > 0.5 and np.median(np.abs(offsets)) <= 10:
            print("VERDICT: yaw is already correct; no residual offset needed.")
        else:
            print("VERDICT: a per-session yaw alignment is required, as the runtime")
            print("         calibrator does. Offsets above are the estimates.")
    return results


def nhc_check(dataset, sessions, limit=10):
    """A correct vehicle frame should show far less lateral than forward acceleration."""
    rule("3. FRAME SANITY: lateral vs forward acceleration spread")
    print("A correct vehicle frame concentrates longitudinal dynamics on forward.\n")
    print(f"{'session':<10} {'std fwd':>9} {'std right':>10} {'ratio':>7} {'corr lat':>9}")
    for session in sessions[:limit]:
        d = load(dataset, session)
        linear = d["acc"] - d["grav"]
        ok = (np.isfinite(linear).all(axis=1) & np.isfinite(d["azimuth"]) &
              np.isfinite(d["pitch"]) & np.isfinite(d["roll"]) &
              np.isfinite(d["heading"]) & np.isfinite(d["lat_g"]) & (d["speed"] > 2.0))
        if ok.sum() < 500:
            continue
        R = rotation_enu_from_phone(d["azimuth"][ok], d["pitch"][ok], d["roll"][ok])
        veh = enu_to_vehicle(np.einsum("nij,nj->ni", R, linear[ok]), d["heading"][ok])
        lat_ref = d["lat_g"][ok] * GRAVITY
        corr_lat = np.corrcoef(veh[:, 1], lat_ref)[0, 1]
        print(f"{session['session_id']:<10} {veh[:,0].std():>9.3f} {veh[:,1].std():>10.3f} "
              f"{veh[:,1].std()/max(veh[:,0].std(),1e-6):>7.3f} {corr_lat:>9.3f}")


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}")
    gravity_check(dataset, sessions)
    yaw_alignment_check(dataset, sessions)
    nhc_check(dataset, sessions)


if __name__ == "__main__":
    main()
