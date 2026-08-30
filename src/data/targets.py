from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset


# ============================================================
# SMARTPHONE RAW SENSOR COLUMNS
# ============================================================

ACCEL_COLUMNS = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
]

GRAVITY_COLUMNS = [
    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",
]

GYRO_COLUMNS = [
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
]


# ============================================================
# VEHICLE REFERENCE COLUMNS
# ============================================================

VELOCITY_COLUMN = "Velocity (km/hr)"
LATITUDE_COLUMN = "Latitude (degrees)"
LONGITUDE_COLUMN = "Longitude (degrees)"
HEADING_COLUMN = "Heading (degrees)"
YAW_RATE_COLUMN = "Yaw Rate (deg/sec)"


# ============================================================
# ROBUST COLUMN RESOLUTION
# ============================================================

def find_column(df, expected):

    if expected in df.columns:
        return expected

    def normalize(value):

        return (
            str(value)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
        )

    expected_norm = normalize(expected)

    for column in df.columns:

        if normalize(column) == expected_norm:
            return column

    raise KeyError(
        f"\nColumn not found: {expected}\n"
        f"\nAvailable columns:\n{list(df.columns)}"
    )


# ============================================================
# GPS → LOCAL EAST/NORTH
# ============================================================

def gps_to_local_xy(latitude, longitude):

    latitude = np.asarray(
        latitude,
        dtype=np.float64
    )

    longitude = np.asarray(
        longitude,
        dtype=np.float64
    )

    earth_radius = 6371000.0

    lat0 = latitude[0]
    lon0 = longitude[0]

    dlat = np.radians(
        latitude - lat0
    )

    dlon = np.radians(
        longitude - lon0
    )

    north = (
        dlat
        * earth_radius
    )

    east = (
        dlon
        * earth_radius
        * np.cos(
            np.radians(lat0)
        )
    )

    return east, north


# ============================================================
# EAST/NORTH → VEHICLE FORWARD/LATERAL
# ============================================================

def global_to_vehicle_frame(
    delta_east,
    delta_north,
    heading_deg
):
    """
    Convert local East/North displacement to
    vehicle-frame displacement.

    Heading convention:
        0°   = North
        90°  = East

    Output:
        forward  = vehicle longitudinal displacement
        lateral  = vehicle lateral displacement
    """

    heading = np.radians(
        heading_deg
    )

    forward = (
        delta_east * np.sin(heading)
        +
        delta_north * np.cos(heading)
    )

    lateral = (
        delta_east * np.cos(heading)
        -
        delta_north * np.sin(heading)
    )

    return forward, lateral


# ============================================================
# MOTION CLASS
# ============================================================

def create_motion_classes(
    speed_mps,
    yaw_rate_rad_s
):

    motion = np.ones(
        len(speed_mps),
        dtype=np.int64
    )

    stationary = (
        np.abs(speed_mps) < 0.5
    )

    turning = (
        np.abs(yaw_rate_rad_s)
        > np.radians(5.0)
    )

    motion[stationary] = 0

    motion[
        turning & ~stationary
    ] = 2

    return motion


# ============================================================
# PREPARE SYNCHRONIZED SESSION
# ============================================================

def prepare_session(
    dataset,
    session
):

    smartphone_df, vehicle_df = (
        dataset.load_session(session)
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    smartphone_df.columns = (
        smartphone_df.columns
        .astype(str)
        .str.strip()
    )

    vehicle_df.columns = (
        vehicle_df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Resolve smartphone columns
    # --------------------------------------------------------

    accel_cols = [
        find_column(
            smartphone_df,
            column
        )
        for column in ACCEL_COLUMNS
    ]

    gravity_cols = [
        find_column(
            smartphone_df,
            column
        )
        for column in GRAVITY_COLUMNS
    ]

    gyro_cols = [
        find_column(
            smartphone_df,
            column
        )
        for column in GYRO_COLUMNS
    ]

    # --------------------------------------------------------
    # Resolve vehicle columns
    # --------------------------------------------------------

    velocity_col = find_column(
        vehicle_df,
        VELOCITY_COLUMN
    )

    latitude_col = find_column(
        vehicle_df,
        LATITUDE_COLUMN
    )

    longitude_col = find_column(
        vehicle_df,
        LONGITUDE_COLUMN
    )

    heading_col = find_column(
        vehicle_df,
        HEADING_COLUMN
    )

    yaw_rate_col = find_column(
        vehicle_df,
        YAW_RATE_COLUMN
    )

    # --------------------------------------------------------
    # Convert smartphone measurements
    # --------------------------------------------------------

    acceleration = smartphone_df[
        accel_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    gravity = smartphone_df[
        gravity_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    gyroscope = smartphone_df[
        gyro_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Remove gravity
    #
    # The IO-VNBD paper explicitly provides gravity
    # measurements to assist acceleration correction.
    # --------------------------------------------------------

    linear_acceleration = (
        acceleration - gravity
    )

    # --------------------------------------------------------
    # 6-channel model input
    #
    # [linear_ax, linear_ay, linear_az,
    #  gyro_yaw, gyro_pitch, gyro_roll]
    # --------------------------------------------------------

    imu = np.concatenate(
        [
            linear_acceleration,
            gyroscope
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Vehicle reference data
    # --------------------------------------------------------

    speed_kmh = pd.to_numeric(
        vehicle_df[velocity_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    latitude = pd.to_numeric(
        vehicle_df[latitude_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    longitude = pd.to_numeric(
        vehicle_df[longitude_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    heading_deg = pd.to_numeric(
        vehicle_df[heading_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    yaw_rate_deg_s = pd.to_numeric(
        vehicle_df[yaw_rate_col],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Unit conversion
    # --------------------------------------------------------

    speed_mps = (
        speed_kmh / 3.6
    )

    yaw_rate_rad_s = np.radians(
        yaw_rate_deg_s
    )

    # --------------------------------------------------------
    # GPS trajectory
    # --------------------------------------------------------

    east, north = gps_to_local_xy(
        latitude,
        longitude
    )

    # --------------------------------------------------------
    # Reference dataframe
    # --------------------------------------------------------

    targets = pd.DataFrame({

        "speed_mps":
            speed_mps,

        "position_east_m":
            east,

        "position_north_m":
            north,

        "heading_deg":
            heading_deg,

        "yaw_rate_rad_s":
            yaw_rate_rad_s,

        "motion_class":
            create_motion_classes(
                speed_mps,
                yaw_rate_rad_s
            )
    })

    # --------------------------------------------------------
    # Synchronization safety
    # --------------------------------------------------------

    n = min(
        len(imu),
        len(targets)
    )

    imu = imu[:n]

    targets = targets.iloc[
        :n
    ].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Remove invalid synchronized samples
    # --------------------------------------------------------

    valid_imu = np.isfinite(
        imu
    ).all(axis=1)

    valid_targets = np.isfinite(
        targets[
            [
                "speed_mps",
                "position_east_m",
                "position_north_m",
                "heading_deg",
                "yaw_rate_rad_s"
            ]
        ].to_numpy(
            dtype=np.float64
        )
    ).all(axis=1)

    valid = (
        valid_imu
        &
        valid_targets
    )

    imu = imu[valid]

    targets = targets.loc[
        valid
    ].reset_index(
        drop=True
    )

    if len(imu) == 0:

        raise ValueError(
            f"Session {session['session_id']} "
            "contains no valid synchronized samples."
        )

    return (
        pd.DataFrame(
            imu,
            columns=[
                "linear_accel_x",
                "linear_accel_y",
                "linear_accel_z",
                "gyro_yaw",
                "gyro_pitch",
                "gyro_roll"
            ]
        ),
        targets
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    dataset = IOVNBDSynchronizedDataset(
        "data/IO-VNBD/synchronized"
    )

    sessions = dataset.get_sessions()

    print("=" * 70)
    print("FINAL IO-VNBD TARGET / INPUT TEST")
    print("=" * 70)

    print(
        f"\nSessions: {len(sessions)}"
    )

    imu, targets = prepare_session(
        dataset,
        sessions[0]
    )

    print(
        f"\nIMU shape: {imu.shape}"
    )

    print(
        f"Target shape: {targets.shape}"
    )

    print(
        "\nModel input columns:"
    )

    for column in imu.columns:
        print(
            f"  {column}"
        )

    print(
        "\nTarget columns:"
    )

    for column in targets.columns:
        print(
            f"  {column}"
        )

    print(
        "\nNaN / Inf:"
    )

    print(
        "IMU:",
        np.isfinite(
            imu.to_numpy()
        ).all()
    )

    print(
        "Targets:",
        np.isfinite(
            targets[
                [
                    "speed_mps",
                    "position_east_m",
                    "position_north_m",
                    "heading_deg",
                    "yaw_rate_rad_s"
                ]
            ].to_numpy()
        ).all()
    )

    print(
        "\nFINAL TARGET PIPELINE TEST PASSED"
    )