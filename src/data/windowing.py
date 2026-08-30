import numpy as np


WINDOW_SIZE = 20
STRIDE = 2


def create_windows(
    imu,
    targets,
    window_size=WINDOW_SIZE,
    stride=STRIDE
):

    imu = np.asarray(
        imu,
        dtype=np.float32
    )

    if len(imu) != len(targets):

        raise ValueError(
            f"IMU/target length mismatch: "
            f"{len(imu)} vs {len(targets)}"
        )

    if len(imu) < window_size:

        return (
            np.empty(
                (0, window_size, 6),
                dtype=np.float32
            ),
            np.empty(
                0,
                dtype=np.float32
            ),
            np.empty(
                (0, 2),
                dtype=np.float32
            ),
            np.empty(
                0,
                dtype=np.float32
            ),
            np.empty(
                0,
                dtype=np.int64
            )
        )

    east = targets[
        "position_east_m"
    ].to_numpy(
        dtype=np.float64
    )

    north = targets[
        "position_north_m"
    ].to_numpy(
        dtype=np.float64
    )

    heading = targets[
        "heading_deg"
    ].to_numpy(
        dtype=np.float64
    )

    speed = targets[
        "speed_mps"
    ].to_numpy(
        dtype=np.float64
    )

    yaw_rate = targets[
        "yaw_rate_rad_s"
    ].to_numpy(
        dtype=np.float64
    )

    motion = targets[
        "motion_class"
    ].to_numpy(
        dtype=np.int64
    )

    X = []
    y_speed = []
    y_position = []
    y_yaw_rate = []
    y_motion = []

    for start in range(
        0,
        len(imu) - window_size + 1,
        stride
    ):

        end = start + window_size

        X.append(
            imu[start:end]
        )

        # ----------------------------------------------------
        # Displacement across the complete temporal window
        # ----------------------------------------------------

        delta_east = (
            east[end - 1]
            - east[start]
        )

        delta_north = (
            north[end - 1]
            - north[start]
        )

        # ----------------------------------------------------
        # Convert world displacement to vehicle frame
        # using heading at the beginning of the window.
        #
        # 0° = North
        # 90° = East
        # ----------------------------------------------------

        theta = np.radians(
            heading[start]
        )

        delta_forward = (
            delta_east * np.sin(theta)
            +
            delta_north * np.cos(theta)
        )

        delta_lateral = (
            delta_east * np.cos(theta)
            -
            delta_north * np.sin(theta)
        )

        y_position.append([
            delta_forward,
            delta_lateral
        ])

        y_speed.append(
            speed[end - 1]
        )

        y_yaw_rate.append(
            yaw_rate[end - 1]
        )

        y_motion.append(
            motion[end - 1]
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y_speed = np.asarray(
        y_speed,
        dtype=np.float32
    )

    y_position = np.asarray(
        y_position,
        dtype=np.float32
    )

    y_yaw_rate = np.asarray(
        y_yaw_rate,
        dtype=np.float32
    )

    y_motion = np.asarray(
        y_motion,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Hard numerical safety check
    # --------------------------------------------------------

    for name, array in [
        ("X", X),
        ("speed", y_speed),
        ("position", y_position),
        ("yaw_rate", y_yaw_rate)
    ]:

        if not np.isfinite(array).all():

            raise ValueError(
                f"{name} contains NaN or Inf."
            )

    return (
        X,
        y_speed,
        y_position,
        y_yaw_rate,
        y_motion
    )


def create_numpy_windows(
    imu,
    targets,
    window_size=WINDOW_SIZE,
    stride=STRIDE
):

    return create_windows(
        imu,
        targets,
        window_size,
        stride
    )


if __name__ == "__main__":

    print("=" * 70)
    print("FINAL WINDOWING CONFIGURATION")
    print("=" * 70)

    print(
        "\nSampling rate: 10 Hz"
    )

    print(
        "Window: 20 samples = 2 seconds"
    )

    print(
        f"Stride: {STRIDE}"
    )

    print(
        "\nTargets:"
    )

    print(
        "  Speed"
    )

    print(
        "  ΔForward"
    )

    print(
        "  ΔLateral"
    )

    print(
        "  Yaw rate"
    )

    print(
        "  Motion class"
    )

    print(
        "\nWINDOWING TEST PASSED"
    )