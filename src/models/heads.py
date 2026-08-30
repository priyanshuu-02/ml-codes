import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    """
    Multi-task prediction heads.

    Outputs:
        speed        -> predicted vehicle speed (m/s)
        position     -> predicted relative displacement (dx, dy) in meters
        yaw_rate     -> predicted yaw rate (rad/s)
        motion       -> stationary / straight / turning
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=64,
        dropout=0.1,
        num_motion_classes=3
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Speed + log variance
        self.speed_head = nn.Linear(
            hidden_dim,
            2
        )

        # Relative position dx, dy + uncertainty
        self.position_head = nn.Linear(
            hidden_dim,
            4
        )

        # Yaw rate + uncertainty
        self.yaw_rate_head = nn.Linear(
            hidden_dim,
            2
        )

        # Motion classification
        self.motion_head = nn.Linear(
            hidden_dim,
            num_motion_classes
        )

    def forward(self, x):
        """
        x:
            [B, D]

        Returns:
            dictionary containing all predictions.
        """

        features = self.shared(x)

        speed_output = self.speed_head(features)

        position_output = self.position_head(features)

        yaw_output = self.yaw_rate_head(features)

        motion_logits = self.motion_head(features)

        # ----------------------------------------------------
        # Regression predictions
        # ----------------------------------------------------

        speed = speed_output[:, 0]

        speed_log_variance = speed_output[:, 1]

        position = position_output[:, :2]

        position_log_variance = position_output[:, 2:]

        yaw_rate = yaw_output[:, 0]

        yaw_rate_log_variance = yaw_output[:, 1]

        return {
            "speed": speed,

            "speed_log_variance":
                speed_log_variance,

            "position":
                position,

            "position_log_variance":
                position_log_variance,

            "yaw_rate":
                yaw_rate,

            "yaw_rate_log_variance":
                yaw_rate_log_variance,

            "motion_logits":
                motion_logits,
        }


if __name__ == "__main__":

    model = PredictionHeads(
        input_dim=128,
        hidden_dim=64,
        dropout=0.1
    )

    x = torch.randn(
        128,
        128
    )

    outputs = model(x)

    print("Input:", x.shape)

    for name, value in outputs.items():
        print(
            f"{name}: {tuple(value.shape)}"
        )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nParameters: {parameters:,}"
    )