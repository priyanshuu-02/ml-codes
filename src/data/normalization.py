import json
from pathlib import Path

import numpy as np


class TrainOnlyNormalizer:

    def __init__(self):

        self.imu_mean = None
        self.imu_std = None

        self.speed_mean = None
        self.speed_std = None

        self.position_mean = None
        self.position_std = None

        self.yaw_mean = None
        self.yaw_std = None

    @staticmethod
    def _safe_std(std):

        return np.where(
            std < 1e-6,
            1.0,
            std
        )

    def fit(
        self,
        X,
        speed,
        position,
        yaw_rate
    ):
        """
        Calculate statistics ONLY from training data.
        """

        self.imu_mean = X.mean(
            axis=(0, 1)
        )

        self.imu_std = self._safe_std(
            X.std(axis=(0, 1))
        )

        self.speed_mean = float(
            speed.mean()
        )

        self.speed_std = float(
            self._safe_std(
                np.array(speed.std())
            )
        )

        self.position_mean = position.mean(
            axis=0
        )

        self.position_std = self._safe_std(
            position.std(axis=0)
        )

        self.yaw_mean = float(
            yaw_rate.mean()
        )

        self.yaw_std = float(
            self._safe_std(
                np.array(yaw_rate.std())
            )
        )

        return self

    def transform_imu(self, X):

        return (
            X - self.imu_mean
        ) / self.imu_std

    def transform_speed(self, speed):

        return (
            speed - self.speed_mean
        ) / self.speed_std

    def transform_position(self, position):

        return (
            position - self.position_mean
        ) / self.position_std

    def transform_yaw(self, yaw_rate):

        return (
            yaw_rate - self.yaw_mean
        ) / self.yaw_std

    def inverse_speed(self, speed):

        return (
            speed * self.speed_std
            + self.speed_mean
        )

    def inverse_position(self, position):

        return (
            position * self.position_std
            + self.position_mean
        )

    def inverse_yaw(self, yaw_rate):

        return (
            yaw_rate * self.yaw_std
            + self.yaw_mean
        )

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {

            "imu_mean":
                self.imu_mean.tolist(),

            "imu_std":
                self.imu_std.tolist(),

            "speed_mean":
                self.speed_mean,

            "speed_std":
                self.speed_std,

            "position_mean":
                self.position_mean.tolist(),

            "position_std":
                self.position_std.tolist(),

            "yaw_mean":
                self.yaw_mean,

            "yaw_std":
                self.yaw_std
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

    @classmethod
    def load(cls, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        normalizer = cls()

        normalizer.imu_mean = np.asarray(
            data["imu_mean"],
            dtype=np.float32
        )

        normalizer.imu_std = np.asarray(
            data["imu_std"],
            dtype=np.float32
        )

        normalizer.speed_mean = float(
            data["speed_mean"]
        )

        normalizer.speed_std = float(
            data["speed_std"]
        )

        normalizer.position_mean = np.asarray(
            data["position_mean"],
            dtype=np.float32
        )

        normalizer.position_std = np.asarray(
            data["position_std"],
            dtype=np.float32
        )

        normalizer.yaw_mean = float(
            data["yaw_mean"]
        )

        normalizer.yaw_std = float(
            data["yaw_std"]
        )

        return normalizer


if __name__ == "__main__":

    print("=" * 70)
    print("TRAIN-ONLY NORMALIZATION TEST")
    print("=" * 70)

    # Synthetic test data
    X = np.random.randn(
        100,
        20,
        6
    ).astype(np.float32)

    speed = np.random.randn(
        100
    ).astype(np.float32)

    position = np.random.randn(
        100,
        2
    ).astype(np.float32)

    yaw = np.random.randn(
        100
    ).astype(np.float32)

    normalizer = TrainOnlyNormalizer()

    normalizer.fit(
        X,
        speed,
        position,
        yaw
    )

    X_norm = normalizer.transform_imu(X)

    speed_norm = normalizer.transform_speed(
        speed
    )

    position_norm = normalizer.transform_position(
        position
    )

    yaw_norm = normalizer.transform_yaw(
        yaw
    )

    print(
        "\nNormalized IMU mean:",
        X_norm.mean()
    )

    print(
        "Normalized IMU std:",
        X_norm.std()
    )

    print(
        "Normalized speed mean:",
        speed_norm.mean()
    )

    print(
        "Normalized position mean:",
        position_norm.mean()
    )

    print(
        "Normalized yaw mean:",
        yaw_norm.mean()
    )

    output = Path(
        "models/preprocessing/normalization_test.json"
    )

    normalizer.save(output)

    print(
        f"\nSaved test statistics to:"
        f"\n{output}"
    )

    print(
        "\nNORMALIZATION TEST SUCCESSFUL"
    )