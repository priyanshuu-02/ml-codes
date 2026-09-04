"""
Simulated GNSS blackout evaluation on the held-out test session.

This is the deployment gate. Per-window error is not what matters for dead reckoning:
small biases compound through integration, which is exactly how the shipped V8 reached
1.85 m position RMSE per window and still drifted 373 m over 60 s.

Protocol, for each blackout start point and each duration:
  1. Seed heading and speed from ground truth, as a real outage would from the last
     trusted GNSS fix.
  2. Integrate forward using only model outputs, feeding the model's own predicted speed
     back in as the next window's initial speed. Open loop, no truth after the seed.
  3. Compare the final integrated position against ground truth.

Windows are chained with stride = window - 1, so consecutive window displacements abut
exactly with neither overlap nor gap. Accumulating overlapping displacements would
multiply distance travelled.

Three systems are compared on identical blackouts:
  IDR-V1      the new model
  V8          the shipped artifact, fed its own contract (gravity retained, its own
              normalisation), so the comparison is fair rather than rigged
  persistence constant speed, no turning. What a constant-velocity INS achieves with no
              network at all, and the bar any model must clear to be worth deploying.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import (
    SAMPLE_RATE_HZ,
    VERIFIED_SESSIONS,
    WINDOW_SAMPLES,
    IdrNormalizer,
    build_windows,
    prepare_session,
)
from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset

DATASET_ROOT = "data/IO-VNBD/synchronized"
CHAIN_STRIDE = WINDOW_SAMPLES - 1  # 19 samples, so windows abut exactly
DURATIONS_SECONDS = [10, 20, 30, 60]


def load_onnx(path, names):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    available = {output.name for output in session.get_outputs()}
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"{path} is missing outputs {missing}")
    return session


class IdrRunner:
    """New model, on the IDR-V1 contract."""

    name = "IDR-V1"

    def __init__(self, model_dir):
        self.normalizer = IdrNormalizer.load(Path(model_dir) / "normalization.json")
        self.outputs = ["speed", "position", "heading_delta"]
        self.session = load_onnx(Path(model_dir) / "idr_v1.onnx", self.outputs)

    def prepare(self, prepared):
        # Gravity already removed by prepare_session, gyro in file order.
        return prepared["imu"]

    def run(self, imu_window, initial_speed_mps):
        imu = self.normalizer.imu(imu_window[None, ...]).astype(np.float32)
        speed = np.asarray(
            [self.normalizer.speed_forward(np.float32(initial_speed_mps))], dtype=np.float32
        )
        speed_out, position_out, heading_out = self.session.run(
            self.outputs, {"imu": imu, "initial_speed_normalized": speed}
        )
        position = self.normalizer.position_inverse(position_out[0])
        return {
            "speed": float(self.normalizer.speed_inverse(speed_out[0])),
            "forward": float(position[0]),
            "lateral": float(position[1]),
            "heading_delta": float(heading_out[0]),
        }


class V8Runner:
    """
    Shipped artifact, on its own contract.

    V8 was trained with gravity present, so the gravity vector is added back before feeding
    it. Its own normalisation file is used. This gives V8 the input distribution it expects.
    """

    name = "V8"

    def __init__(self, assets_dir):
        assets_dir = Path(assets_dir)
        stats = json.loads((assets_dir / "v8_normalization.json").read_text(encoding="utf-8"))
        self.imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        self.imu_std = np.asarray(stats["imu_std"], dtype=np.float32)
        self.speed_mean = float(stats["speed_mean"])
        self.speed_std = float(stats["speed_std"])
        self.position_mean = np.asarray(stats["position_mean"], dtype=np.float32)
        self.position_std = np.asarray(stats["position_std"], dtype=np.float32)
        self.outputs = ["speed", "position", "heading_delta"]
        self.session = load_onnx(assets_dir / "v8_dead_reckoning.onnx", self.outputs)

    def prepare(self, prepared):
        imu = prepared["imu"].copy()
        imu[:, 0:3] += prepared["gravity"]
        return imu

    def run(self, imu_window, initial_speed_mps):
        imu = ((imu_window - self.imu_mean) / self.imu_std)[None, ...].astype(np.float32)
        speed = np.asarray(
            [(initial_speed_mps - self.speed_mean) / self.speed_std], dtype=np.float32
        )
        speed_out, position_out, heading_out = self.session.run(
            self.outputs, {"imu": imu, "initial_speed_normalized": speed}
        )
        position = position_out[0] * self.position_std + self.position_mean
        return {
            "speed": float(speed_out[0] * self.speed_std + self.speed_mean),
            "forward": float(position[0]),
            "lateral": float(position[1]),
            "heading_delta": float(heading_out[0]),
        }


class PersistenceRunner:
    """Constant speed, no turning. No network."""

    name = "persistence"

    def __init__(self, span_seconds):
        self.span = span_seconds

    def prepare(self, prepared):
        return prepared["imu"]

    def run(self, imu_window, initial_speed_mps):
        return {
            "speed": float(initial_speed_mps),
            "forward": float(initial_speed_mps * self.span),
            "lateral": 0.0,
            "heading_delta": 0.0,
        }


def integrate(runner, imu_source, windows, truth, start_index, window_count):
    """
    Roll a blackout forward, open loop.

    Displacement is rotated by the START-of-window heading and heading is advanced after,
    matching VehicleFusionEkf.predict and the target definition.
    """
    heading = np.radians(truth["heading"][windows["start"][start_index]])
    east = truth["east"][windows["start"][start_index]]
    north = truth["north"][windows["start"][start_index]]
    speed = float(truth["speed"][windows["start"][start_index]])

    for offset in range(window_count):
        index = start_index + offset
        prediction = runner.run(imu_source[index], speed)

        rotation = heading
        north += prediction["forward"] * np.cos(rotation) - prediction["lateral"] * np.sin(rotation)
        east += prediction["forward"] * np.sin(rotation) + prediction["lateral"] * np.cos(rotation)
        heading += prediction["heading_delta"]
        # Feeding the model's own speed back is what a real outage does.
        speed = max(prediction["speed"], 0.0)

    end_sample = windows["end"][start_index + window_count - 1]
    return east, north, end_sample


def evaluate(session_id, runners, samples_per_start=90, seed=0):
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    available = {s["session_id"]: s for s in dataset.get_sessions()}
    session = available[session_id]
    prepared = prepare_session(dataset, session, VERIFIED_SESSIONS[session_id]["lag"])
    windows = build_windows(prepared, window=WINDOW_SAMPLES, stride=CHAIN_STRIDE)
    if windows is None:
        raise RuntimeError(f"{session_id}: no chainable windows")

    prepared_inputs = {runner.name: runner.prepare(prepared) for runner in runners}
    window_inputs = {
        name: np.stack([array[s:s + WINDOW_SAMPLES] for s in windows["start"]])
        for name, array in prepared_inputs.items()
    }

    span = (WINDOW_SAMPLES - 1) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(seed)
    results = {runner.name: {} for runner in runners}

    print(f"\ntest session {session_id}: {len(windows['start']):,} chainable windows "
          f"({CHAIN_STRIDE / SAMPLE_RATE_HZ:.1f} s each)")

    for duration in DURATIONS_SECONDS:
        window_count = int(round(duration / span))
        if window_count < 1 or len(windows["start"]) <= window_count + 1:
            continue

        limit = len(windows["start"]) - window_count - 1
        # Only start blackouts where the vehicle is actually moving; a blackout while parked
        # is trivially easy and would flatter every system.
        candidates = [
            index for index in range(limit)
            if prepared["speed"][windows["start"][index]] > 3.0
        ]
        if len(candidates) < 10:
            continue
        starts = rng.choice(candidates, size=min(samples_per_start, len(candidates)), replace=False)

        for runner in runners:
            errors, drifts = [], []
            for start_index in starts:
                east, north, end_sample = integrate(
                    runner, window_inputs[runner.name], windows, prepared, int(start_index), window_count
                )
                truth_east = prepared["east"][end_sample]
                truth_north = prepared["north"][end_sample]
                error = float(np.hypot(east - truth_east, north - truth_north))

                begin_sample = windows["start"][start_index]
                path = float(np.sum(np.hypot(
                    np.diff(prepared["east"][begin_sample:end_sample + 1]),
                    np.diff(prepared["north"][begin_sample:end_sample + 1]),
                )))
                errors.append(error)
                if path > 1.0:
                    drifts.append(error / path * 100.0)

            results[runner.name][duration] = {
                "blackouts": len(starts),
                "median_error_m": float(np.median(errors)),
                "mean_error_m": float(np.mean(errors)),
                "p90_error_m": float(np.percentile(errors, 90)),
                "median_drift_pct": float(np.median(drifts)) if drifts else float("nan"),
                "mean_drift_pct": float(np.mean(drifts)) if drifts else float("nan"),
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="Y1")
    parser.add_argument("--model-dir", default="models/deploy/idr_v1")
    parser.add_argument("--v8-dir", default="../app/src/main/assets/ml")
    parser.add_argument("--blackouts", type=int, default=90)
    args = parser.parse_args()

    span = (WINDOW_SAMPLES - 1) / SAMPLE_RATE_HZ
    runners = [IdrRunner(args.model_dir)]

    v8_dir = Path(args.v8_dir)
    if (v8_dir / "v8_dead_reckoning.onnx").exists():
        try:
            runners.append(V8Runner(v8_dir))
        except Exception as error:  # noqa: BLE001 - report and continue without V8
            print(f"V8 unavailable for comparison: {error}")
    runners.append(PersistenceRunner(span))

    results = evaluate(args.session, runners, samples_per_start=args.blackouts)

    print("\n" + "=" * 92)
    print("MEDIAN FINAL POSITION ERROR AND DRIFT, BY BLACKOUT DURATION")
    print("=" * 92)
    header = f"{'system':<14}" + "".join(f"{f'{d}s err':>11}{f'{d}s drift':>11}" for d in DURATIONS_SECONDS)
    print(header)
    print("-" * len(header))
    for name, per_duration in results.items():
        row = f"{name:<14}"
        for duration in DURATIONS_SECONDS:
            entry = per_duration.get(duration)
            if entry is None:
                row += f"{'-':>11}{'-':>11}"
            else:
                row += f"{entry['median_error_m']:>10.1f}m{entry['median_drift_pct']:>10.1f}%"
        print(row)

    print("\nSIH target is under 10 percent drift.")
    for name, per_duration in results.items():
        verdicts = []
        for duration in DURATIONS_SECONDS:
            entry = per_duration.get(duration)
            if entry is None:
                continue
            verdicts.append(f"{duration}s {'PASS' if entry['median_drift_pct'] < 10 else 'FAIL'}")
        print(f"  {name:<14} {'  '.join(verdicts)}")

    output = Path("outputs/blackout_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"session": args.session, "results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
