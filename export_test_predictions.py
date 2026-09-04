"""
Dump everything the Kotlin ablation test needs for the held-out session.

Why this exists rather than a Python re-implementation of the estimator:

The thing being measured in Stage 8 is the *full pipeline*, which means the real
VehicleFusionEkf, the real non-holonomic constraint, the real HiddenMarkovRoadMatcher
and the real anisotropic map update. Those live in Kotlin. Porting them to Python to
run an evaluation would measure a copy, and any divergence between the copy and the
shipped code would silently invalidate the result. That is the same class of mistake
that produced the V8 artifact, whose training preprocessing and runtime preprocessing
disagreed without anyone noticing.

So the split is: Python owns ground truth and calibration, Kotlin owns inference and the
estimator. This script writes the boundary between them into the JVM test's resources.

Two files are produced:

  predictions.json  per-window ground truth, the gyro calibration, the road proxy, and a
                    reference set of Python-side predictions used purely as a parity check
                    against the JVM's own inference.
  samples.bin       float32 [N, 9]: linear acceleration xyz, gyro xyz, gravity xyz.

The raw samples are exported rather than finished predictions because the model consumes
its own previous speed estimate. Precomputing predictions would mean seeding every window
from ground-truth speed, which leaks truth into every step of a blackout and flatters the
result. The JVM test therefore runs the shipped ONNX artifact itself and closes the speed
loop through the estimator, which is what actually happens on the phone.

Windows are emitted with stride = WINDOW_SAMPLES - 1, so consecutive window
displacements abut exactly. Any smaller stride would double-count distance when the
displacements are accumulated.

Three things in here are worth treating with suspicion, and are labelled as such in
the output file so a reader cannot miss them:

  1. The road polyline is synthesised from this session's own ground truth, decimated
     and noised. Real OSM geometry has its own error and the vehicle is not on the
     centreline. The map-matching stage of the ablation is therefore OPTIMISTIC.
  2. The gyro scale factor is fitted on this session against the vehicle's own yaw
     rate. That mirrors what a runtime calibrator does while GNSS is available, and
     the fit is held fixed through the blackout, but it is still session-specific.
  3. n = 1 test session. Y1 is the only held-out session with verified IMU-to-vehicle
     correspondence, so nothing here supports a claim about population variance.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.idr_dataset import (
    CHANNEL_NAMES,
    SAMPLE_RATE_HZ,
    VERIFIED_SESSIONS,
    VERSION,
    WINDOW_SAMPLES,
    WINDOW_SPAN_SECONDS,
    IdrNormalizer,
    build_windows,
    prepare_session,
    verify_correspondence,
)
from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset

DATASET_ROOT = "data/IO-VNBD/synchronized"
CHAIN_STRIDE = WINDOW_SAMPLES - 1

# Road synthesis. Spacing is loosely OSM-like; the noise is a stand-in for geometry error.
ROAD_SPACING_METERS = 10.0
ROAD_NOISE_SIGMA_METERS = 3.0
ROAD_POINTS_PER_WAY = 40
ROAD_SEED = 20260831

EARTH_RADIUS_METERS = 6371000.0


# --- model runners ------------------------------------------------------------

class IdrRunner:
    """
    IDR-V1 on its own contract, including the log-variance heads.

    Sigma is converted exactly as IdrMotionEngine.sigma does on Android
    (exp(0.5 * clamp(logvar, -20, 10)) then scaled into engineering units), so the
    uncertainty the Kotlin test feeds to updateSpeed is the same number the phone
    would compute.
    """

    name = "idr"

    OUTPUTS = [
        "speed", "speed_log_variance",
        "acceleration",
        "position", "position_log_variance",
        "heading_delta", "heading_delta_log_variance",
    ]

    def __init__(self, model_dir):
        import onnxruntime

        model_dir = Path(model_dir)
        self.normalizer = IdrNormalizer.load(model_dir / "normalization.json")
        self.session = onnxruntime.InferenceSession(
            str(model_dir / "idr_v1.onnx"), providers=["CPUExecutionProvider"]
        )
        available = {output.name for output in self.session.get_outputs()}
        missing = [name for name in self.OUTPUTS if name not in available]
        if missing:
            raise RuntimeError(f"idr_v1.onnx is missing outputs {missing}")

    def inputs(self, prepared):
        return prepared["imu"]

    @staticmethod
    def _sigma(log_variance):
        return np.exp(0.5 * np.clip(log_variance, -20.0, 10.0))

    def run_batch(self, imu_windows, initial_speeds):
        imu = self.normalizer.imu(imu_windows).astype(np.float32)
        speed = self.normalizer.speed_forward(initial_speeds.astype(np.float32))
        (
            speed_out, speed_logvar,
            acceleration_out,
            position_out, position_logvar,
            heading_out, heading_logvar,
        ) = self.session.run(self.OUTPUTS, {"imu": imu, "initial_speed_normalized": speed})

        position = self.normalizer.position_inverse(position_out)
        position_sigma = self._sigma(position_logvar) * self.normalizer.position_scale
        return {
            "forward": position[:, 0],
            "lateral": position[:, 1],
            "heading_delta": heading_out,
            "speed": np.maximum(self.normalizer.speed_inverse(speed_out), 0.0),
            "speed_sigma": self._sigma(speed_logvar) * self.normalizer.speed_std,
            "forward_sigma": position_sigma[:, 0],
            "lateral_sigma": position_sigma[:, 1],
            "heading_sigma": self._sigma(heading_logvar),
            "acceleration": self.normalizer.acceleration_inverse(acceleration_out),
        }


class V8Runner:
    """
    Shipped artifact on its own contract, so the comparison is fair rather than rigged.

    V8 was trained with gravity present, so gravity is added back before feeding it, and
    its own normalisation file is used. It has no trained variance heads, so no sigma is
    exported for it; the Kotlin side falls back to a fixed speed uncertainty for config A.
    """

    name = "v8"

    OUTPUTS = ["speed", "position", "heading_delta"]

    def __init__(self, assets_dir):
        import onnxruntime

        assets_dir = Path(assets_dir)
        stats = json.loads((assets_dir / "v8_normalization.json").read_text(encoding="utf-8"))
        self.imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        self.imu_std = np.asarray(stats["imu_std"], dtype=np.float32)
        self.speed_mean = float(stats["speed_mean"])
        self.speed_std = float(stats["speed_std"])
        self.position_mean = np.asarray(stats["position_mean"], dtype=np.float32)
        self.position_std = np.asarray(stats["position_std"], dtype=np.float32)
        self.session = onnxruntime.InferenceSession(
            str(assets_dir / "v8_dead_reckoning.onnx"), providers=["CPUExecutionProvider"]
        )

    def inputs(self, prepared):
        imu = prepared["imu"].copy()
        imu[:, 0:3] += prepared["gravity"]
        return imu

    def run_batch(self, imu_windows, initial_speeds):
        imu = ((imu_windows - self.imu_mean) / self.imu_std).astype(np.float32)
        speed = ((initial_speeds - self.speed_mean) / self.speed_std).astype(np.float32)
        speed_out, position_out, heading_out = self.session.run(
            self.OUTPUTS, {"imu": imu, "initial_speed_normalized": speed}
        )
        position = position_out * self.position_std + self.position_mean
        return {
            "forward": position[:, 0],
            "lateral": position[:, 1],
            "heading_delta": heading_out,
            "speed": np.maximum(speed_out * self.speed_std + self.speed_mean, 0.0),
        }


# --- gyro calibration ---------------------------------------------------------

def calibrate_gyro(prepared):
    """
    Fit a single scale factor mapping one phone gyro channel onto vehicle yaw rate.

    Only one channel is used, deliberately. The three gyro columns in this dataset are
    not a rotatable vector: rotating the triple into the vehicle frame gives correlation
    0.035 against the vehicle's yaw rate while the single best channel gives 0.95, and
    all 48 axis permutations and sign combinations were measured to fail. They behave
    like Euler rates, not body rates. So the honest treatment is to pick the channel that
    is the yaw-rate channel and scale it.

    The fit is least squares through the origin, which also recovers the sign. It uses
    the vehicle's own yaw rate as reference, which at runtime corresponds to calibrating
    while GNSS is healthy and then holding the result through the outage.
    """
    valid = prepared["valid"]
    gyro = prepared["imu"][valid, 3:6]
    reference = prepared["yaw_rate"][valid]

    best = {"channel": None, "scale": 1.0, "correlation": 0.0}
    for index in range(gyro.shape[1]):
        channel = gyro[:, index]
        if np.std(channel) < 1e-9:
            continue
        correlation = float(np.corrcoef(channel, reference)[0, 1])
        if abs(correlation) > abs(best["correlation"]):
            denominator = float(np.dot(channel, channel))
            best = {
                "channel": index,
                "scale": float(np.dot(channel, reference) / denominator) if denominator > 0 else 1.0,
                "correlation": correlation,
            }
    if best["channel"] is None:
        raise RuntimeError("no usable gyro channel")

    calibrated = prepared["imu"][:, 3 + best["channel"]] * best["scale"]

    # Report how well integrated heading tracks truth over a fixed horizon, because that
    # is what actually matters during an outage and it is far less forgiving than the
    # instantaneous correlation.
    horizon = 10 * SAMPLE_RATE_HZ
    truth_heading = np.radians(prepared["heading"])
    integrated = np.concatenate([[0.0], np.cumsum(calibrated[:-1] / SAMPLE_RATE_HZ)])
    if len(integrated) > horizon:
        gyro_change = integrated[horizon:] - integrated[:-horizon]
        truth_change = truth_heading[horizon:] - truth_heading[:-horizon]
        moving = prepared["speed"][:-horizon] > 3.0
        finite = np.isfinite(gyro_change) & np.isfinite(truth_change) & moving
        drift = float(np.median(np.abs(gyro_change[finite] - truth_change[finite]))) if finite.any() else float("nan")
    else:
        drift = float("nan")

    return {
        "channel": best["channel"],
        "channel_name": CHANNEL_NAMES[3 + best["channel"]],
        "scale": best["scale"],
        "correlation": best["correlation"],
        "median_heading_error_10s_deg": float(np.degrees(drift)),
    }, calibrated


# --- road synthesis -----------------------------------------------------------

def synthesize_road(latitude, longitude, spacing, noise_sigma, seed):
    """
    Build an OSM-like polyline from the driven path.

    This is a PROXY, and an optimistic one. A real road network is not derived from the
    vehicle's own trajectory: its geometry error is uncorrelated with the drive, roads
    have lanes so the vehicle sits offset from the centreline, and nearby parallel roads
    create genuine ambiguity that this construction cannot reproduce. Decimating and
    adding noise makes the geometry imperfect but does not make it independent.

    Reported map-matching gains are therefore an upper bound, not an expectation.
    """
    rng = np.random.default_rng(seed)
    scale_north = EARTH_RADIUS_METERS * np.pi / 180.0
    scale_east = scale_north * np.cos(np.radians(latitude[0]))

    keep = [0]
    last_lat, last_lon = latitude[0], longitude[0]
    for index in range(1, len(latitude)):
        north = (latitude[index] - last_lat) * scale_north
        east = (longitude[index] - last_lon) * scale_east
        if np.hypot(east, north) >= spacing:
            keep.append(index)
            last_lat, last_lon = latitude[index], longitude[index]
    if keep[-1] != len(latitude) - 1:
        keep.append(len(latitude) - 1)

    points = np.column_stack([latitude[keep], longitude[keep]])

    # Correlated noise: geometry error in a real network varies smoothly along a way
    # rather than jittering node to node.
    raw = rng.normal(0.0, noise_sigma, size=(len(points), 2))
    kernel = np.array([0.25, 0.5, 0.25])
    smoothed = np.column_stack([
        np.convolve(raw[:, axis], kernel, mode="same") / np.sqrt((kernel ** 2).sum())
        for axis in range(2)
    ])
    points[:, 0] += smoothed[:, 0] / scale_north
    points[:, 1] += smoothed[:, 1] / scale_east
    return points


def split_into_ways(points, per_way):
    """
    Chop the polyline into overlapping ways, so the matcher sees a graph with transitions
    rather than one enormous segment. Overlap by one node keeps the ways connected.
    """
    ways = []
    start = 0
    while start < len(points) - 1:
        end = min(start + per_way, len(points) - 1)
        ways.append({
            "way_id": len(ways) + 1,
            "name": f"Route segment {len(ways) + 1}",
            "points": [[round(float(lat), 7), round(float(lon), 7)] for lat, lon in points[start:end + 1]],
        })
        start = end
    return ways


# --- assembly -----------------------------------------------------------------

def build(session_id, model_dir, v8_dir, output_path, road_spacing, road_noise):
    dataset = IOVNBDSynchronizedDataset(DATASET_ROOT)
    available = {s["session_id"]: s for s in dataset.get_sessions()}
    if session_id not in available:
        raise RuntimeError(f"session {session_id} not found in {DATASET_ROOT}")
    if session_id not in VERIFIED_SESSIONS:
        raise RuntimeError(f"session {session_id} has no verified IMU-to-vehicle correspondence")

    meta = VERIFIED_SESSIONS[session_id]
    prepared = prepare_session(dataset, available[session_id], meta["lag"])

    achieved, problem = verify_correspondence(prepared, meta["corr"])
    if problem is not None:
        raise RuntimeError(f"{session_id}: {problem}")
    print(f"{session_id}: {len(prepared['imu']):,} samples, correspondence {abs(achieved):.3f} "
          f"(expected {meta['corr']:.3f})")

    windows = build_windows(prepared, window=WINDOW_SAMPLES, stride=CHAIN_STRIDE)
    if windows is None:
        raise RuntimeError(f"{session_id}: no chainable windows")
    starts, ends = windows["start"], windows["end"]
    print(f"{len(starts):,} chainable windows at stride {CHAIN_STRIDE} "
          f"({WINDOW_SPAN_SECONDS:.1f} s each)")

    calibration, calibrated_gyro = calibrate_gyro(prepared)
    print(f"gyro: channel {calibration['channel']} ({calibration['channel_name']}) "
          f"scale {calibration['scale']:+.4f} corr {calibration['correlation']:+.3f} "
          f"median heading error over 10 s {calibration['median_heading_error_10s_deg']:.2f} deg")

    runners = [IdrRunner(model_dir)]
    v8_path = Path(v8_dir)
    if (v8_path / "v8_dead_reckoning.onnx").exists():
        try:
            runners.append(V8Runner(v8_path))
        except Exception as error:  # noqa: BLE001 - a missing baseline must not stop the export
            print(f"V8 baseline unavailable: {error}")
    else:
        print(f"V8 baseline not found under {v8_path}")

    predictions = {}
    initial_speeds = prepared["speed"][starts].astype(np.float32)
    for runner in runners:
        source = runner.inputs(prepared)
        batch = np.stack([source[s:s + WINDOW_SAMPLES] for s in starts]).astype(np.float32)
        predictions[runner.name] = runner.run_batch(batch, initial_speeds)
        print(f"{runner.name}: {len(starts):,} windows inferred")

    # Ground-truth path length per window, needed for drift as a percentage of distance.
    step_east = np.diff(prepared["east"])
    step_north = np.diff(prepared["north"])
    step_length = np.hypot(step_east, step_north)
    cumulative = np.concatenate([[0.0], np.cumsum(step_length)])
    path_lengths = cumulative[ends] - cumulative[starts]

    # Reconstruct latitude/longitude. prepare_session only keeps local East/North, and the
    # Kotlin side works in GeoPoint, so both endpoints are converted back. Position error
    # in the test is computed geodesically to avoid depending on either side's flat-earth
    # constant.
    smartphone, vehicle = dataset.load_session(available[session_id])
    vehicle.columns = vehicle.columns.astype(str).str.strip()
    from src.data.columns import resolve
    import pandas as pd

    def column(name):
        return pd.to_numeric(vehicle[resolve(vehicle, name)], errors="coerce").to_numpy(dtype=np.float64)

    full_latitude, full_longitude = column("Latitude"), column("Longitude")
    shift = -meta["lag"]
    if shift > 0:
        aligned_latitude, aligned_longitude = full_latitude[shift:], full_longitude[shift:]
    elif shift < 0:
        aligned_latitude, aligned_longitude = full_latitude[:shift], full_longitude[:shift]
    else:
        aligned_latitude, aligned_longitude = full_latitude, full_longitude
    length = len(prepared["east"])
    latitude = aligned_latitude[:length]
    longitude = aligned_longitude[:length]
    if len(latitude) != length or not np.isfinite(latitude).all():
        raise RuntimeError("latitude reconstruction does not line up with the prepared arrays")

    # Cross-check the reconstruction against the East/North the pipeline actually used.
    scale_north = EARTH_RADIUS_METERS * np.pi / 180.0
    scale_east = scale_north * np.cos(np.radians(latitude[0]))
    residual = np.max(np.abs((latitude - latitude[0]) * scale_north - prepared["north"]))
    residual = max(residual, np.max(np.abs((longitude - longitude[0]) * scale_east - prepared["east"])))
    if residual > 0.5:
        raise RuntimeError(f"latitude/longitude reconstruction disagrees by {residual:.3f} m")
    print(f"latitude/longitude reconstruction agrees with East/North to {residual:.4f} m")

    heading = np.mod(prepared["heading"], 360.0)

    road_points = synthesize_road(latitude, longitude, road_spacing, road_noise, ROAD_SEED)
    ways = split_into_ways(road_points, ROAD_POINTS_PER_WAY)
    print(f"road proxy: {len(road_points):,} nodes in {len(ways):,} ways "
          f"({road_spacing:.0f} m spacing, {road_noise:.0f} m correlated noise)")

    records = []
    reference_names = {"idr": "idrRef", "v8": "v8Ref"}
    for index in range(len(starts)):
        start, end = int(starts[index]), int(ends[index])
        gyro_slice = calibrated_gyro[start:end]  # 19 intervals across a 20-sample window
        record = {
            "start": start,
            "end": end,
            "startLat": round(float(latitude[start]), 7),
            "startLon": round(float(longitude[start]), 7),
            "startHeadingDeg": round(float(heading[start]), 4),
            "startSpeedMps": round(float(prepared["speed"][start]), 4),
            "endLat": round(float(latitude[end]), 7),
            "endLon": round(float(longitude[end]), 7),
            "endHeadingDeg": round(float(heading[end]), 4),
            "endSpeedMps": round(float(prepared["speed"][end]), 4),
            "pathMeters": round(float(path_lengths[index]), 4),
            "gyroRates": [round(float(value), 5) for value in gyro_slice],
        }
        for name, batch in predictions.items():
            entry = {
                "forward": round(float(batch["forward"][index]), 4),
                "lateral": round(float(batch["lateral"][index]), 4),
                "headingDelta": round(float(batch["heading_delta"][index]), 6),
                "speed": round(float(batch["speed"][index]), 4),
            }
            if "speed_sigma" in batch:
                entry["speedSigma"] = round(float(batch["speed_sigma"][index]), 4)
                entry["headingSigma"] = round(float(batch["heading_sigma"][index]), 6)
            record[reference_names[name]] = entry
        records.append(record)

    payload = {
        "session": session_id,
        "contract": {
            "preprocessing_version": VERSION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_samples": WINDOW_SAMPLES,
            "chain_stride": CHAIN_STRIDE,
            "window_span_seconds": WINDOW_SPAN_SECONDS,
            "lag_samples": meta["lag"],
            "correspondence": round(float(abs(achieved)), 4),
        },
        "limitations": [
            "The road polyline is synthesised from this session's own ground truth, "
            "decimated and noised. It is an optimistic proxy for real OSM geometry, so "
            "any map-matching improvement is an upper bound.",
            "The gyro scale factor is fitted on this session against the vehicle's own "
            "yaw rate, mirroring runtime calibration during GNSS availability. It is "
            "held fixed through every simulated blackout but is session-specific.",
            "n = 1 test session. Y1 is the only held-out session with verified "
            "IMU-to-vehicle correspondence, so nothing here bounds population variance.",
        ],
        "gyroCalibration": {
            "channel": calibration["channel"],
            "channelName": calibration["channel_name"],
            "scale": round(calibration["scale"], 6),
            "correlation": round(calibration["correlation"], 4),
            "medianHeadingError10sDeg": round(calibration["median_heading_error_10s_deg"], 4),
        },
        "road": {
            "note": "OPTIMISTIC PROXY. Derived from this session's ground truth, not from OSM.",
            "spacingMeters": road_spacing,
            "noiseSigmaMeters": road_noise,
            "seed": ROAD_SEED,
            "ways": ways,
        },
        "samples": {
            "file": "samples.bin",
            "dtype": "float32",
            "order": "row-major, big-endian",
            "count": int(len(prepared["imu"])),
            "channels": [
                "lin_accel_x", "lin_accel_y", "lin_accel_z",
                "gyro_ch0", "gyro_ch1", "gyro_ch2",
                "gravity_x", "gravity_y", "gravity_z",
            ],
            "note": "Channels 0-5 are the IDR-V1 input directly. V8 expects gravity "
                    "present, so its accelerometer input is channels 0-2 plus 6-8.",
        },
        "referenceNote": "idrRef and v8Ref were produced in Python seeded from "
                         "ground-truth initial speed. They exist only so the JVM can "
                         "assert inference parity. The ablation must not use them, "
                         "because seeding from truth every window leaks truth into the "
                         "blackout.",
        "windows": records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"wrote {output_path} ({size_mb:.1f} MiB, {len(records):,} windows)")

    # Big-endian to match java.io.DataInputStream, so the JVM side needs no byte juggling.
    samples = np.concatenate([prepared["imu"], prepared["gravity"]], axis=1).astype(">f4")
    if samples.shape[1] != 9:
        raise RuntimeError(f"expected 9 channels, built {samples.shape[1]}")
    samples_path = output_path.parent / "samples.bin"
    samples_path.write_bytes(samples.tobytes())
    print(f"wrote {samples_path} ({samples_path.stat().st_size / (1024 * 1024):.1f} MiB, "
          f"{samples.shape[0]:,} samples x {samples.shape[1]} channels)")

    # Sanity check the exported displacements before anything downstream trusts them.
    for name, batch in predictions.items():
        forward, lateral = batch["forward"], batch["lateral"]
        implied = float(np.mean(initial_speeds)) * WINDOW_SPAN_SECONDS
        print(f"  {name}: forward mean {forward.mean():+.3f} m (speed implies {implied:.3f} m), "
              f"lateral mean {lateral.mean():+.3f} m, "
              f"lateral/forward std ratio {lateral.std() / max(forward.std(), 1e-9):.3f}")

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="Y1")
    parser.add_argument("--model-dir", default="models/deploy/idr_v1")
    parser.add_argument("--v8-dir", default="../app/src/main/assets/ml")
    parser.add_argument(
        "--output", default="../app/src/test/resources/blackout/predictions.json"
    )
    parser.add_argument("--road-spacing", type=float, default=ROAD_SPACING_METERS)
    parser.add_argument("--road-noise", type=float, default=ROAD_NOISE_SIGMA_METERS)
    args = parser.parse_args()

    build(
        args.session,
        args.model_dir,
        args.v8_dir,
        args.output,
        args.road_spacing,
        args.road_noise,
    )


if __name__ == "__main__":
    main()
