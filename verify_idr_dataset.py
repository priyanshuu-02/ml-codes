"""
Verify the IDR-V1 dataset before anything is trained on it.

The previous model shipped with normalisation statistics that were physically impossible,
and nothing in the pipeline objected. This script is the gate that would have caught it:
it builds the split, reports the displacement statistics, and refuses to pass if they are
not physically consistent with the measured speeds.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import (
    WINDOW_SPAN_SECONDS,
    build_split,
    check_plausibility,
)

SHIPPED = {
    "speed_mean": 11.62447452545166,
    "position_mean": [-2.996354579925537, 3.694854974746704],
    "position_std": [19.846437454223633, 18.584957122802734],
}


def main():
    print("Building IDR-V1 split from verified sessions only\n")
    splits, normalizer = build_split(verbose=True)

    failures = []
    print()
    for name in ("train", "validation", "test"):
        data = splits.get(name)
        if data is None:
            print(f"--- {name}: EMPTY")
            failures.append(f"{name} split is empty")
            continue

        forward = data["position"][:, 0]
        lateral = data["position"][:, 1]
        speed = data["speed"]
        implied = speed.mean() * WINDOW_SPAN_SECONDS

        print(f"--- {name}: {len(data['imu']):,} windows, "
              f"sessions {sorted(set(data['session_id'].tolist()))}")
        print(f"    speed mean     {speed.mean():8.3f} m/s")
        print(f"    accel mean     {data['acceleration'].mean():8.3f} m/s^2  "
              f"(std {data['acceleration'].std():.3f})")
        print(f"    forward mean   {forward.mean():8.3f} m   "
              f"(speed x {WINDOW_SPAN_SECONDS}s implies {implied:.3f})")
        print(f"    lateral mean   {lateral.mean():8.3f} m")
        print(f"    forward std    {forward.std():8.3f} m")
        print(f"    lateral std    {lateral.std():8.3f} m")
        ratio = lateral.std() / max(forward.std(), 1e-9)
        print(f"    lat/fwd ratio  {ratio:8.3f}   (must stay below 0.35)")
        print(f"    heading delta  mean {data['heading_delta'].mean():+.4f} rad  "
              f"std {data['heading_delta'].std():.4f}")
        counts = np.bincount(data["motion"], minlength=3).tolist()
        print(f"    motion classes stationary={counts[0]:,} straight={counts[1]:,} turning={counts[2]:,}")

        reasons = check_plausibility(speed, forward, lateral)
        if reasons:
            print(f"    PLAUSIBILITY   FAIL")
            for reason in reasons:
                print(f"                   - {reason}")
            failures.append(f"{name}: {'; '.join(reasons)}")
        else:
            print(f"    PLAUSIBILITY   PASS")
        print()

    print("=" * 78)
    print("NORMALISATION (train split only)")
    print("=" * 78)
    stats = normalizer.to_dict()
    for index, channel in enumerate(stats["channel_names"]):
        print(f"  {channel:<14} mean {stats['imu_mean'][index]:+9.4f}  std {stats['imu_std'][index]:8.4f}")
    print(f"  speed          mean {stats['speed_mean']:+9.4f}  std {stats['speed_std']:8.4f}")
    print(f"  acceleration   mean {stats['acceleration_mean']:+9.4f}  std {stats['acceleration_std']:8.4f}")
    print(f"  position scale {stats['position_scale']}  (mean fixed at {stats['position_mean']})")

    print()
    print("Gravity check: lin_accel_z mean should be near 0, not near 9.79.")
    print(f"  lin_accel_z mean = {stats['imu_mean'][2]:+.4f}   "
          f"(shipped artifact had +9.7896, gravity left in)")

    print()
    print("=" * 78)
    print("COMPARISON WITH THE SHIPPED ARTIFACT")
    print("=" * 78)
    shipped_ratio = SHIPPED["position_std"][1] / SHIPPED["position_std"][0]
    train = splits["train"]
    new_ratio = train["position"][:, 1].std() / train["position"][:, 0].std()
    print(f"  {'':<22} {'shipped':>12} {'IDR-V1':>12}")
    print(f"  {'forward mean (m)':<22} {SHIPPED['position_mean'][0]:>12.3f} "
          f"{train['position'][:,0].mean():>12.3f}")
    print(f"  {'lateral mean (m)':<22} {SHIPPED['position_mean'][1]:>12.3f} "
          f"{train['position'][:,1].mean():>12.3f}")
    print(f"  {'lat/fwd std ratio':<22} {shipped_ratio:>12.3f} {new_ratio:>12.3f}")
    print(f"  {'verdict':<22} {'IMPOSSIBLE':>12} "
          f"{'PLAUSIBLE' if new_ratio < 0.35 else 'IMPOSSIBLE':>12}")

    output = Path("outputs/idr_v1_normalization.json")
    normalizer.save(output)
    print(f"\nwrote {output}")

    print()
    if failures:
        print("RESULT: FAILED. Not safe to train.")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: PASSED. Dataset is physically consistent and safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
