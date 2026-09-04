"""
IDR-V1 dataset: the contract the Android runtime actually implements.

This module deliberately replaces the older windowing path rather than extending it,
because several of its assumptions were measured to be false:

  * Sessions were paired by exact filename case, silently discarding 40 of 72.
  * The gyro columns were treated as (yaw, pitch, roll). Correlated against the vehicle's
    own yaw rate, the column labelled Pitch reaches 0.95 while Yaw sits at 0.00.
  * Every session was assumed synchronised. Only 6 of 72 have smartphone IMU that tracks
    the vehicle log, and most need a per-session lag correction first.
  * Gravity handling disagreed between the shipped normalisation (mean Z 9.79, gravity
    present) and the training code (gravity subtracted).

Only sessions whose IMU demonstrably corresponds to their ground truth are used, each
shifted by its measured lag. Correspondence is verified with the phone gyro against the
vehicle's reported yaw rate, which is instantaneous in both streams and needs no frame.

Mirrors PreprocessingSpec.IDR_V1 on the Android side, field for field.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.columns import resolve
from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset

# --- Contract. Must stay identical to PreprocessingSpec.IDR_V1 -----------------

VERSION = "idr-v1"
SAMPLE_RATE_HZ = 10
WINDOW_SAMPLES = 20
STRIDE_SAMPLES = 2
CHANNEL_NAMES = [
    "lin_accel_x", "lin_accel_y", "lin_accel_z",
    "gyro_ch0", "gyro_ch1", "gyro_ch2",
]
CHANNEL_COUNT = len(CHANNEL_NAMES)
WINDOW_SPAN_SECONDS = (WINDOW_SAMPLES - 1) / SAMPLE_RATE_HZ  # 1.9 s, not 2.0

GRAVITY = 9.80665

# --- Session selection --------------------------------------------------------

# Correspondence measured by diagnose_sync.py: |corr| between the best phone gyro channel
# and the vehicle's Yaw Rate, after applying the per-session lag. Lag is in samples and is
# applied as "advance the vehicle stream by `lag`".
VERIFIED_SESSIONS = {
    "S3c": {"lag": -5, "corr": 0.996},
    "S3a": {"lag": 67, "corr": 0.974},
    "S1": {"lag": -2, "corr": 0.948},
    "S2": {"lag": -87, "corr": 0.919},
    "Y1": {"lag": -1157, "corr": 0.800},
    "M": {"lag": -13, "corr": 0.778},
}

# Held out entirely. Y1 is a different driver and vehicle from the training sessions, which
# makes it the most honest available test of generalisation.
TEST_SESSIONS = ["Y1"]
VALIDATION_SESSIONS = ["S3a"]

BORDERLINE_SESSIONS = {
    "S3b": {"lag": -47, "corr": 0.459},
    "S4": {"lag": -18, "corr": 0.334},
}


def session_plan(include_borderline=False):
    """Which sessions to use and how each splits."""
    chosen = dict(VERIFIED_SESSIONS)
    if include_borderline:
        chosen.update(BORDERLINE_SESSIONS)

    plan = {"train": [], "validation": [], "test": []}
    for session_id, meta in chosen.items():
        if session_id in TEST_SESSIONS:
            plan["test"].append(session_id)
        elif session_id in VALIDATION_SESSIONS:
            plan["validation"].append(session_id)
        else:
            plan["train"].append(session_id)
    return plan, chosen


# --- Per-session extraction ---------------------------------------------------

def _numeric(frame, name):
    return pd.to_numeric(frame[resolve(frame, name)], errors="coerce").to_numpy(dtype=np.float64)


def prepare_session(dataset, session, lag_samples):
    """
    Build the IDR-V1 channels and reference signals for one session.

    Gravity is removed using the dataset's own GRAVITY column, so the result is true linear
    acceleration in phone axes. The gyro columns are passed through in file order.

    `lag_samples` is the offset measured by diagnose_sync.py, whose cross-correlation
    convention means

        gyro[t]  corresponds to  vehicle[t - lag]

    so the shift applied below uses the NEGATED lag. Getting this backwards doubles the
    misalignment instead of removing it, which is silent: the arrays still line up in shape
    and the displacement targets still look perfectly plausible, because those are built
    from vehicle columns alone. It showed up only when integrated gyro heading was compared
    against the vehicle's own yaw rate, where S2 read -0.008 and Y1 read 0.031 against
    measured values of 0.919 and 0.800. [verify_correspondence] now guards it.
    """
    smartphone, vehicle = dataset.load_session(session)
    smartphone.columns = smartphone.columns.astype(str).str.strip()
    vehicle.columns = vehicle.columns.astype(str).str.strip()

    n = min(len(smartphone), len(vehicle))

    accel = np.column_stack([_numeric(smartphone, f"ACCELEROMETER {ax}") for ax in "XYZ"])[:n]
    gravity = np.column_stack([_numeric(smartphone, f"GRAVITY {ax}") for ax in "XYZ"])[:n]
    gyro = np.column_stack([
        _numeric(smartphone, "GYROSCOPE Yaw"),
        _numeric(smartphone, "GYROSCOPE Pitch"),
        _numeric(smartphone, "GYROSCOPE Roll"),
    ])[:n]
    imu = np.concatenate([accel - gravity, gyro], axis=1)

    latitude = _numeric(vehicle, "Latitude")[:n]
    longitude = _numeric(vehicle, "Longitude")[:n]
    heading = _numeric(vehicle, "Heading")[:n]
    speed = (_numeric(vehicle, "Velocity") / 3.6)[:n]
    yaw_rate = np.radians(_numeric(vehicle, "Yaw Rate")[:n])
    longitudinal = _numeric(vehicle, "Indicated Longitudinal Acceleration")[:n] * GRAVITY

    # gyro[t] corresponds to vehicle[t - lag], so shift by the negated lag.
    shift = -lag_samples
    gravity_aligned = gravity
    if shift > 0:
        imu = imu[: n - shift]
        gravity_aligned = gravity[: n - shift]
        latitude, longitude = latitude[shift:], longitude[shift:]
        heading, speed = heading[shift:], speed[shift:]
        yaw_rate, longitudinal = yaw_rate[shift:], longitudinal[shift:]
    elif shift < 0:
        advance = -shift
        imu = imu[advance:]
        gravity_aligned = gravity[advance:]
        latitude, longitude = latitude[: n - advance], longitude[: n - advance]
        heading, speed = heading[: n - advance], speed[: n - advance]
        yaw_rate, longitudinal = yaw_rate[: n - advance], longitudinal[: n - advance]

    length = min(len(imu), len(latitude))
    imu = imu[:length]
    gravity_aligned = gravity_aligned[:length]
    latitude, longitude = latitude[:length], longitude[:length]
    heading, speed = heading[:length], speed[:length]
    yaw_rate, longitudinal = yaw_rate[:length], longitudinal[:length]

    # Local East/North, matching the runtime's flat-earth handling.
    earth = 6371000.0
    east = np.radians(longitude - longitude[0]) * earth * np.cos(np.radians(latitude[0]))
    north = np.radians(latitude - latitude[0]) * earth

    # Unwrap heading so window differences never wrap the wrong way.
    heading_unwrapped = np.degrees(np.unwrap(np.radians(heading)))

    valid = (
        np.isfinite(imu).all(axis=1)
        & np.isfinite(east) & np.isfinite(north)
        & np.isfinite(heading_unwrapped) & np.isfinite(speed) & np.isfinite(yaw_rate)
        & np.isfinite(longitudinal)
    )

    return {
        "imu": imu,
        # Retained so an evaluation can reconstruct the gravity-present input that the
        # shipped V8 artifact expects, making the comparison fair rather than rigged.
        "gravity": gravity_aligned,
        "east": east,
        "north": north,
        "heading": heading_unwrapped,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "longitudinal": longitudinal,
        "valid": valid,
    }


def build_windows(prepared, window=WINDOW_SAMPLES, stride=STRIDE_SAMPLES):
    """
    Window the session into IDR-V1 samples.

    Displacement is expressed in the vehicle frame as it was at the START of the window,
    matching both VehicleFusionEkf.predict and the verified target convention:

        forward = dEast sin(h) + dNorth cos(h)
        lateral = dEast cos(h) - dNorth sin(h)
    """
    imu = prepared["imu"]
    east, north = prepared["east"], prepared["north"]
    heading, speed = prepared["heading"], prepared["speed"]
    yaw_rate, longitudinal = prepared["yaw_rate"], prepared["longitudinal"]
    valid = prepared["valid"]

    total = len(imu)
    if total < window:
        return None

    starts = np.arange(0, total - window + 1, stride, dtype=np.int64)
    ends = starts + window - 1
    if len(starts) == 0:
        return None

    # Drop any window containing a non-finite sample.
    cumulative = np.concatenate([[0], np.cumsum(~valid)])
    bad = cumulative[ends + 1] - cumulative[starts]
    keep = bad == 0
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return None

    delta_east = east[ends] - east[starts]
    delta_north = north[ends] - north[starts]
    theta = np.radians(heading[starts])
    forward = delta_east * np.sin(theta) + delta_north * np.cos(theta)
    lateral = delta_east * np.cos(theta) - delta_north * np.sin(theta)

    heading_delta = np.radians(heading[ends] - heading[starts])

    # Motion class, matching the previous definition so results stay comparable.
    motion = np.ones(len(starts), dtype=np.int64)
    motion[speed[ends] < 0.5] = 0
    motion[(np.abs(yaw_rate[ends]) > np.radians(5.0)) & (speed[ends] >= 0.5)] = 2

    windows = np.stack([imu[s:s + window] for s in starts]).astype(np.float32)

    return {
        "imu": windows,
        # Endpoint speed is what the estimator needs for velocity propagation.
        "speed": speed[ends].astype(np.float32),
        "initial_speed": speed[starts].astype(np.float32),
        # Acceleration is an explicit SIH requirement the previous model never predicted.
        "acceleration": longitudinal[ends].astype(np.float32),
        "position": np.column_stack([forward, lateral]).astype(np.float32),
        "heading_delta": heading_delta.astype(np.float32),
        "motion": motion,
        "start": starts,
        "end": ends,
    }


# --- Normalisation ------------------------------------------------------------

class IdrNormalizer:
    """
    Train-split statistics only.

    Displacement is normalised by scale alone, with the mean left at zero. Centring
    displacement was what allowed the previous artifact to ship statistics implying a
    negative mean forward travel: a model regressing toward its target mean then emits a
    systematic backwards-and-sideways bias every window, which integrates into the
    trajectory. Zero displacement must mean zero motion.
    """

    def __init__(self):
        self.imu_mean = None
        self.imu_std = None
        self.speed_mean = None
        self.speed_std = None
        self.acceleration_mean = None
        self.acceleration_std = None
        self.position_scale = None

    @staticmethod
    def _safe(std):
        return np.where(np.asarray(std) < 1e-6, 1.0, std)

    def fit(self, imu, speed, acceleration, position):
        self.imu_mean = imu.mean(axis=(0, 1))
        self.imu_std = self._safe(imu.std(axis=(0, 1)))
        self.speed_mean = float(speed.mean())
        self.speed_std = float(self._safe(np.array(speed.std())))
        self.acceleration_mean = float(acceleration.mean())
        self.acceleration_std = float(self._safe(np.array(acceleration.std())))
        self.position_scale = self._safe(position.std(axis=0))
        return self

    def imu(self, values):
        return (values - self.imu_mean) / self.imu_std

    def speed_forward(self, values):
        return (values - self.speed_mean) / self.speed_std

    def speed_inverse(self, values):
        return values * self.speed_std + self.speed_mean

    def acceleration_forward(self, values):
        return (values - self.acceleration_mean) / self.acceleration_std

    def acceleration_inverse(self, values):
        return values * self.acceleration_std + self.acceleration_mean

    def position_forward(self, values):
        return values / self.position_scale

    def position_inverse(self, values):
        return values * self.position_scale

    def to_dict(self):
        return {
            "preprocessing_version": VERSION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_samples": WINDOW_SAMPLES,
            "stride_samples": STRIDE_SAMPLES,
            "window_span_seconds": WINDOW_SPAN_SECONDS,
            "channel_names": CHANNEL_NAMES,
            "frame": "PHONE_LINEAR",
            "gravity": "REMOVED",
            "gyro_order": "DATASET_NATIVE",
            "imu_mean": np.asarray(self.imu_mean).tolist(),
            "imu_std": np.asarray(self.imu_std).tolist(),
            "speed_mean": self.speed_mean,
            "speed_std": self.speed_std,
            "acceleration_mean": self.acceleration_mean,
            "acceleration_std": self.acceleration_std,
            # Zero-centred by construction; kept explicit so the runtime cannot assume
            # otherwise.
            "position_mean": [0.0, 0.0],
            "position_scale": np.asarray(self.position_scale).tolist(),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        normalizer = cls()
        normalizer.imu_mean = np.asarray(data["imu_mean"], dtype=np.float32)
        normalizer.imu_std = np.asarray(data["imu_std"], dtype=np.float32)
        normalizer.speed_mean = float(data["speed_mean"])
        normalizer.speed_std = float(data["speed_std"])
        normalizer.acceleration_mean = float(data["acceleration_mean"])
        normalizer.acceleration_std = float(data["acceleration_std"])
        normalizer.position_scale = np.asarray(data["position_scale"], dtype=np.float32)
        return normalizer


# --- Plausibility gate -------------------------------------------------------

def check_plausibility(speed, forward, lateral, span_seconds=WINDOW_SPAN_SECONDS):
    """
    Mirror of DisplacementStatsGuard on the Android side.

    Encodes the failure that shipped last time: statistics implying 35 km/h of sideways
    travel and a negative mean forward displacement. Raising here means the pipeline is
    wrong, and nothing should be trained on it.
    """
    reasons = []
    expected_forward = float(speed.mean()) * span_seconds
    if forward.mean() <= 0 < expected_forward:
        reasons.append(
            f"forward mean {forward.mean():.3f} m is not positive while mean speed "
            f"{speed.mean():.3f} m/s over {span_seconds} s implies {expected_forward:.3f} m"
        )
    elif expected_forward > 0:
        relative = abs(forward.mean() - expected_forward) / expected_forward
        if relative > 0.45:
            reasons.append(
                f"forward mean {forward.mean():.3f} m disagrees with speed-implied "
                f"{expected_forward:.3f} m by {relative*100:.1f}%"
            )
    if forward.std() > 0:
        ratio = lateral.std() / forward.std()
        if ratio > 0.35:
            reasons.append(f"lateral/forward std ratio {ratio:.3f} exceeds 0.35")
    implied_lateral_speed = lateral.std() / span_seconds
    if implied_lateral_speed > 3.0:
        reasons.append(f"lateral 1-sigma implies {implied_lateral_speed*3.6:.1f} km/h sideways")
    if abs(lateral.mean()) > 2.0:
        reasons.append(f"lateral mean {lateral.mean():.3f} m should be near zero")
    return reasons


# --- Assembly ----------------------------------------------------------------

def verify_correspondence(prepared, expected_corr, tolerance=0.25):
    """
    Confirm the smartphone IMU really does line up with the vehicle log after shifting.

    Correlates each raw gyro channel against the vehicle's reported yaw rate and returns the
    strongest. Both are instantaneous rotation measurements, so this needs no frame, no
    calibration and no model.

    This guard exists because a sign error in the lag shift is otherwise invisible: array
    shapes still match, and the displacement targets still pass every plausibility check,
    since those are derived from vehicle columns alone and never touch the IMU.
    """
    valid = prepared["valid"]
    if valid.sum() < 1000:
        return None, "too few valid samples to verify"

    gyro = prepared["imu"][valid, 3:6]
    reference = prepared["yaw_rate"][valid]
    if np.std(reference) < 1e-3:
        return None, "too little turning to verify"

    best = 0.0
    for index in range(3):
        channel = gyro[:, index]
        if np.std(channel) < 1e-9:
            continue
        correlation = float(np.corrcoef(channel, reference)[0, 1])
        if abs(correlation) > abs(best):
            best = correlation

    if abs(best) < abs(expected_corr) - tolerance:
        return best, (
            f"correspondence {abs(best):.3f} is far below the measured {abs(expected_corr):.3f}; "
            "the lag shift is probably applied in the wrong direction"
        )
    return best, None


def build_split(dataset_root="data/IO-VNBD/synchronized", include_borderline=False, verbose=True):
    """Build train / validation / test arrays plus a fitted normaliser."""
    dataset = IOVNBDSynchronizedDataset(dataset_root)
    available = {s["session_id"]: s for s in dataset.get_sessions()}
    plan, chosen = session_plan(include_borderline)

    splits = {}
    for split_name, session_ids in plan.items():
        collected = []
        for session_id in session_ids:
            session = available.get(session_id)
            if session is None:
                if verbose:
                    print(f"  [{split_name}] {session_id}: MISSING, skipped")
                continue
            prepared = prepare_session(dataset, session, chosen[session_id]["lag"])

            achieved, problem = verify_correspondence(prepared, chosen[session_id]["corr"])
            if problem is not None:
                raise RuntimeError(f"{session_id}: {problem}")

            windows = build_windows(prepared)
            if windows is None:
                if verbose:
                    print(f"  [{split_name}] {session_id}: no usable windows")
                continue
            windows["session_id"] = np.array([session_id] * len(windows["imu"]))
            collected.append(windows)
            if verbose:
                achieved_text = "n/a" if achieved is None else f"{abs(achieved):.3f}"
                print(f"  [{split_name}] {session_id}: {len(windows['imu']):,} windows "
                      f"(lag {chosen[session_id]['lag']:+d}, expected corr "
                      f"{chosen[session_id]['corr']:.3f}, achieved {achieved_text})")
        if not collected:
            splits[split_name] = None
            continue
        keys = [k for k in collected[0] if k != "session_id"]
        merged = {k: np.concatenate([c[k] for c in collected]) for k in keys}
        merged["session_id"] = np.concatenate([c["session_id"] for c in collected])
        splits[split_name] = merged

    train = splits.get("train")
    if train is None:
        raise RuntimeError("No training windows were produced.")

    normalizer = IdrNormalizer().fit(
        train["imu"], train["speed"], train["acceleration"], train["position"]
    )
    return splits, normalizer
