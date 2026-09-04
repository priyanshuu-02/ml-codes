"""
Empirically establish the IO-VNBD preprocessing contract.

The shipped V8 artifact's own normalisation statistics are physically impossible as
vehicle-frame displacement:

    speed_mean    = 11.624 m/s over a 1.9 s window  =>  forward mean should be ~ +22 m
    position_mean = [-3.00, +3.70] m                =>  forward mean was NEGATIVE
    position_std  = [19.85, 18.59] m                =>  lateral/forward std ratio 0.936
    lateral 1-sigma = 18.59 m in 1.9 s              =>  35 km/h sideways, impossible

A road vehicle does not travel 18 m sideways in two seconds, and forward/lateral
displacement is not isotropic. Something upstream of training was wrong. This script
determines what, from the data itself, rather than guessing.

Nothing here trains or writes model artifacts. It only measures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset
from src.data.targets import find_column

DATASET_ROOT = "data/IO-VNBD/synchronized"

SHIPPED = {
    "speed_mean": 11.62447452545166,
    "position_mean": [-2.996354579925537, 3.694854974746704],
    "position_std": [19.846437454223633, 18.584957122802734],
}


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def describe_rates(dataset, sessions, limit=8):
    """Are the smartphone and vehicle streams actually sample-aligned?"""
    rule("1. SAMPLE RATES AND ROW ALIGNMENT")
    print(f"{'session':<10} {'S rows':>8} {'V rows':>8} {'S Hz':>7} {'V Hz':>7} "
          f"{'S dur':>8} {'V dur':>8} {'ratio':>6}")
    rows = []
    for session in sessions[:limit]:
        smartphone, vehicle = dataset.load_session(session)
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        s_time_col = find_column(smartphone, "TIME SINCE START (ms)")
        s_time = pd.to_numeric(smartphone[s_time_col], errors="coerce").to_numpy() / 1000.0

        v_time_col = find_column(vehicle, "Time Since Start of Day (seconds)")
        v_time = pd.to_numeric(vehicle[v_time_col], errors="coerce").to_numpy()

        s_duration = np.nanmax(s_time) - np.nanmin(s_time)
        v_duration = np.nanmax(v_time) - np.nanmin(v_time)
        s_hz = (len(smartphone) - 1) / s_duration if s_duration > 0 else float("nan")
        v_hz = (len(vehicle) - 1) / v_duration if v_duration > 0 else float("nan")

        print(f"{session['session_id']:<10} {len(smartphone):>8} {len(vehicle):>8} "
              f"{s_hz:>7.2f} {v_hz:>7.2f} {s_duration:>8.1f} {v_duration:>8.1f} "
              f"{len(smartphone)/max(len(vehicle),1):>6.2f}")
        rows.append((s_hz, v_hz, len(smartphone), len(vehicle)))
    return rows


def heading_convention(dataset, sessions, limit=6):
    """
    Recover the heading convention from geometry.

    Course over ground from consecutive fixes is unambiguous:
        course = atan2(dEast, dNorth), 0 = North, 90 = East.
    Comparing that with the Heading column reveals the convention actually used.
    """
    rule("2. HEADING CONVENTION, RECOVERED FROM GEOMETRY")
    print("Comparing the Heading column against course computed from lat/lon motion.")
    print("A near-zero median offset means heading is 0=North clockwise.\n")
    print(f"{'session':<10} {'n':>7} {'median off':>11} {'mean|off|':>10} {'corr':>7}")

    offsets = []
    for session in sessions[:limit]:
        _, vehicle = dataset.load_session(session)
        vehicle.columns = vehicle.columns.astype(str).str.strip()

        lat = pd.to_numeric(vehicle[find_column(vehicle, "Latitude (degrees)")], errors="coerce").to_numpy()
        lon = pd.to_numeric(vehicle[find_column(vehicle, "Longitude (degrees)")], errors="coerce").to_numpy()
        heading = pd.to_numeric(vehicle[find_column(vehicle, "Heading (degrees)")], errors="coerce").to_numpy()
        speed = pd.to_numeric(vehicle[find_column(vehicle, "Velocity (km/hr)")], errors="coerce").to_numpy() / 3.6

        earth = 6371000.0
        lat0 = np.nanmean(lat)
        east = np.radians(lon - lon[0]) * earth * np.cos(np.radians(lat0))
        north = np.radians(lat - lat[0]) * earth

        # Only judge while genuinely moving; course is meaningless at rest.
        step = 10
        d_east = east[step:] - east[:-step]
        d_north = north[step:] - north[:-step]
        moving = (speed[:-step] > 5.0) & (np.hypot(d_east, d_north) > 5.0)
        if moving.sum() < 50:
            continue

        course = np.degrees(np.arctan2(d_east[moving], d_north[moving])) % 360.0
        reported = heading[:-step][moving] % 360.0
        offset = (course - reported + 180.0) % 360.0 - 180.0

        correlation = np.corrcoef(np.unwrap(np.radians(course)), np.unwrap(np.radians(reported)))[0, 1]
        print(f"{session['session_id']:<10} {moving.sum():>7} {np.median(offset):>11.2f} "
              f"{np.mean(np.abs(offset)):>10.2f} {correlation:>7.3f}")
        offsets.append(np.median(offset))

    if offsets:
        print(f"\nmedian offset across sessions: {np.median(offsets):+.2f} deg")
        if abs(np.median(offsets)) < 10:
            print("VERDICT: heading is 0 = North, clockwise. The convention in targets.py is correct.")
        else:
            print("VERDICT: heading column does NOT follow 0 = North clockwise.")
    return offsets


def displacement_statistics(dataset, sessions, window=20, stride=2, limit=20):
    """
    Reproduce the training target exactly as windowing.py builds it, then test the
    physical plausibility of the result.
    """
    rule("3. DISPLACEMENT TARGET STATISTICS, AS TRAINING BUILDS THEM")

    def build(align_by_time):
        forwards, laterals, speeds = [], [], []
        for session in sessions[:limit]:
            smartphone, vehicle = dataset.load_session(session)
            smartphone.columns = smartphone.columns.astype(str).str.strip()
            vehicle.columns = vehicle.columns.astype(str).str.strip()

            lat = pd.to_numeric(vehicle[find_column(vehicle, "Latitude (degrees)")], errors="coerce").to_numpy()
            lon = pd.to_numeric(vehicle[find_column(vehicle, "Longitude (degrees)")], errors="coerce").to_numpy()
            heading = pd.to_numeric(vehicle[find_column(vehicle, "Heading (degrees)")], errors="coerce").to_numpy()
            speed = pd.to_numeric(vehicle[find_column(vehicle, "Velocity (km/hr)")], errors="coerce").to_numpy() / 3.6

            valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(heading) & np.isfinite(speed)
            lat, lon, heading, speed = lat[valid], lon[valid], heading[valid], speed[valid]
            if len(lat) < window + 2:
                continue

            if align_by_time:
                # Vehicle stream resampled to the 10 Hz the contract declares.
                v_time = pd.to_numeric(
                    vehicle[find_column(vehicle, "Time Since Start of Day (seconds)")], errors="coerce"
                ).to_numpy()[valid]
                grid = np.arange(v_time[0], v_time[-1], 0.1)
                if len(grid) < window + 2:
                    continue
                lat = np.interp(grid, v_time, lat)
                lon = np.interp(grid, v_time, lon)
                speed = np.interp(grid, v_time, speed)
                heading = np.degrees(np.unwrap(np.radians(np.interp(grid, v_time, heading))))

            earth = 6371000.0
            east = np.radians(lon - lon[0]) * earth * np.cos(np.radians(lat[0]))
            north = np.radians(lat - lat[0]) * earth

            starts = np.arange(0, len(east) - window + 1, stride)
            ends = starts + window - 1
            d_east = east[ends] - east[starts]
            d_north = north[ends] - north[starts]
            theta = np.radians(heading[starts])

            forwards.append(d_east * np.sin(theta) + d_north * np.cos(theta))
            laterals.append(d_east * np.cos(theta) - d_north * np.sin(theta))
            speeds.append(speed[starts])

        return (np.concatenate(forwards), np.concatenate(laterals), np.concatenate(speeds))

    for label, align in (("AS-IS (row index, no resampling)", False),
                         ("TIME-ALIGNED (vehicle resampled to 10 Hz)", True)):
        forward, lateral, speed = build(align)
        span = (window - 1) / 10.0
        print(f"\n--- {label} ---")
        print(f"  windows                 : {len(forward):,}")
        print(f"  speed mean              : {speed.mean():8.3f} m/s")
        print(f"  forward mean            : {forward.mean():8.3f} m   "
              f"(speed x {span}s implies {speed.mean()*span:.2f})")
        print(f"  lateral mean            : {lateral.mean():8.3f} m")
        print(f"  forward std             : {forward.std():8.3f} m")
        print(f"  lateral std             : {lateral.std():8.3f} m")
        ratio = lateral.std() / forward.std() if forward.std() else float("nan")
        print(f"  lateral/forward std     : {ratio:8.3f}   (road vehicle: well below 0.35)")
        print(f"  implied lateral speed   : {lateral.std()/span*3.6:8.1f} km/h sideways at 1 sigma")
        verdict = "PLAUSIBLE" if ratio < 0.35 and forward.mean() > 0 else "IMPOSSIBLE"
        print(f"  VERDICT                 : {verdict}")

    print(f"\n--- SHIPPED ARTIFACT, for comparison ---")
    sr = SHIPPED["position_std"][1] / SHIPPED["position_std"][0]
    print(f"  forward mean            : {SHIPPED['position_mean'][0]:8.3f} m")
    print(f"  lateral mean            : {SHIPPED['position_mean'][1]:8.3f} m")
    print(f"  forward std             : {SHIPPED['position_std'][0]:8.3f} m")
    print(f"  lateral std             : {SHIPPED['position_std'][1]:8.3f} m")
    print(f"  lateral/forward std     : {sr:8.3f}")
    print(f"  VERDICT                 : IMPOSSIBLE")


def imu_channel_check(dataset, sessions, limit=6):
    """Which accelerometer convention matches the shipped normalisation?"""
    rule("4. IMU CHANNELS: GRAVITY HANDLING")
    print("Shipped imu_mean[2] = 9.7896, which is gravity. Training code subtracts")
    print("gravity, so the shipped stats and the shipped training code disagree.\n")
    print(f"{'session':<10} {'raw az':>9} {'lin az':>9} {'raw |a|':>9} {'lin |a|':>9}")
    for session in sessions[:limit]:
        smartphone, _ = dataset.load_session(session)
        smartphone.columns = smartphone.columns.astype(str).str.strip()
        acc = np.column_stack([
            pd.to_numeric(smartphone[find_column(smartphone, f"ACCELEROMETER {ax} (m/s²)")], errors="coerce")
            for ax in "XYZ"
        ])
        grav = np.column_stack([
            pd.to_numeric(smartphone[find_column(smartphone, f"GRAVITY {ax} (m/s²)")], errors="coerce")
            for ax in "XYZ"
        ])
        lin = acc - grav
        print(f"{session['session_id']:<10} {np.nanmean(acc[:, 2]):>9.3f} {np.nanmean(lin[:, 2]):>9.3f} "
              f"{np.nanmean(np.linalg.norm(acc, axis=1)):>9.3f} "
              f"{np.nanmean(np.linalg.norm(lin, axis=1)):>9.3f}")


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    sessions = dataset.get_sessions()
    print(f"Sessions available: {len(sessions)}")

    describe_rates(dataset, sessions)
    heading_convention(dataset, sessions)
    displacement_statistics(dataset, sessions)
    imu_channel_check(dataset, sessions)


if __name__ == "__main__":
    main()
