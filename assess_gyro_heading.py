"""
If the network cannot predict speed, can integrated gyro still carry heading?

The learnability assessment produced a clean negative result for the regression targets:
the model ties or loses to persistence on speed, acceleration, displacement and heading,
both across sessions and within a single session. So the accelerometer-to-speed path is not
usable from this dataset.

Heading is a different matter, and the distinction is physical rather than statistical. A
gyroscope measures yaw rate directly, needing one integration to reach heading, whereas
speed requires integrating accelerometer data through an unknown mount orientation with
gravity leakage. The earlier sweep already measured the phone gyro tracking the vehicle's
reported yaw rate at 0.78 to 0.996 correlation on the verified sessions.

This quantifies what that is worth for dead reckoning: after a per-session scale and sign
fit, which is exactly what the runtime alignment calibrator learns from GNSS course, how
far does integrated gyro heading drift over 10, 20, 30 and 60 seconds?

Measures only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import VERIFIED_SESSIONS, prepare_session
from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset

DATASET_ROOT = "data/IO-VNBD/synchronized"
SAMPLE_RATE_HZ = 10
HORIZONS_SECONDS = [10, 20, 30, 60]


def main():
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    available = {s["session_id"]: s for s in dataset.get_sessions()}

    print("Per-session gyro channel selection and scale fit, then heading drift.\n")
    header = (f"{'session':<8} {'ch':>3} {'scale':>7} {'corr':>6} "
              + " ".join(f"{h:>4}s" for h in HORIZONS_SECONDS))
    print(header)
    print("-" * len(header))

    summary = {}
    aggregate = {horizon: [] for horizon in HORIZONS_SECONDS}

    for session_id, meta in VERIFIED_SESSIONS.items():
        session = available.get(session_id)
        if session is None:
            continue
        prepared = prepare_session(dataset, session, meta["lag"])
        valid = prepared["valid"]
        if valid.sum() < 5000:
            continue

        gyro = prepared["imu"][:, 3:6]
        truth_yaw_rate = prepared["yaw_rate"]

        # Pick the channel and fit a scale, which absorbs mount orientation and sign. This
        # mirrors what VehicleAlignmentCalibrator learns on device from GNSS course.
        best = (None, 0.0, 0.0)
        for index in range(3):
            channel = gyro[valid, index]
            if channel.std() < 1e-9:
                continue
            correlation = float(np.corrcoef(channel, truth_yaw_rate[valid])[0, 1])
            if abs(correlation) > abs(best[2]):
                scale = float(np.dot(channel, truth_yaw_rate[valid]) / np.dot(channel, channel))
                best = (index, scale, correlation)

        index, scale, correlation = best
        if index is None:
            continue

        estimated_rate = gyro[:, index] * scale
        dt = 1.0 / SAMPLE_RATE_HZ

        row = []
        session_result = {"channel": index, "scale": scale, "correlation": correlation}
        for horizon in HORIZONS_SECONDS:
            steps = horizon * SAMPLE_RATE_HZ
            if len(estimated_rate) <= steps + 1:
                row.append(float("nan"))
                continue

            # Drift over every window of this length, using cumulative integrals so the
            # comparison covers the whole session rather than a hand-picked stretch.
            estimated_heading = np.cumsum(np.where(valid, estimated_rate, 0.0)) * dt
            truth_heading = np.cumsum(np.where(valid, truth_yaw_rate, 0.0)) * dt

            estimated_change = estimated_heading[steps:] - estimated_heading[:-steps]
            truth_change = truth_heading[steps:] - truth_heading[:-steps]

            # Only score windows that are fully valid. A cumulative difference over
            # [steps:] - [:-steps] covers original indices i+1..i+steps, so the validity
            # prefix sums must be offset by one to line up with it.
            cumulative_valid = np.concatenate([[0], np.cumsum(~valid)])
            window_bad = cumulative_valid[steps + 1:] - cumulative_valid[1:-steps]
            keep = window_bad == 0
            if keep.sum() < 100:
                row.append(float("nan"))
                continue

            error_degrees = np.degrees(np.abs(estimated_change[keep] - truth_change[keep]))
            median_error = float(np.median(error_degrees))
            row.append(median_error)
            session_result[f"{horizon}s_median_heading_error_deg"] = median_error
            aggregate[horizon].append(median_error)

        print(f"{session_id:<8} {index:>3} {scale:>7.3f} {correlation:>6.3f} "
              + " ".join(f"{value:>5.1f}" for value in row))
        summary[session_id] = session_result

    print("\n" + "=" * 78)
    print("MEDIAN HEADING ERROR FROM INTEGRATED GYRO (degrees)")
    print("=" * 78)
    for horizon in HORIZONS_SECONDS:
        values = aggregate[horizon]
        if values:
            print(f"  {horizon:>3}s outage : median across sessions {np.median(values):6.2f}  "
                  f"worst {np.max(values):6.2f}")

    print()
    print("Why this matters: cross-track position error grows as distance travelled times")
    print("heading error. At 10 m/s over 30 s the vehicle covers 300 m, so each degree of")
    print("heading error costs roughly 5 m of cross-track position.")
    for horizon in HORIZONS_SECONDS:
        values = aggregate[horizon]
        if not values:
            continue
        distance = 10.0 * horizon
        implied = distance * np.radians(np.median(values))
        print(f"  {horizon:>3}s at 10 m/s : {distance:5.0f} m travelled, "
              f"{np.median(values):5.2f} deg -> ~{implied:6.1f} m cross-track "
              f"({100*implied/distance:4.1f}% drift)")

    output = Path("outputs/gyro_heading.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
