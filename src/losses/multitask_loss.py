import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        speed_weight=1.0,
        position_weight=2.0,
        yaw_weight=1.0,
        motion_weight=0.5
    ):
        super().__init__()

        self.speed_weight = speed_weight
        self.position_weight = position_weight
        self.yaw_weight = yaw_weight
        self.motion_weight = motion_weight

        self.huber = nn.SmoothL1Loss(
            beta=1.0
        )

    def forward(
        self,
        outputs,
        targets
    ):

        speed_loss = self.huber(
            outputs["speed"],
            targets["speed"]
        )

        position_loss = self.huber(
            outputs["position"],
            targets["position"]
        )

        yaw_loss = self.huber(
            outputs["yaw_rate"],
            targets["yaw_rate"]
        )

        motion_loss = F.cross_entropy(
            outputs["motion_logits"],
            targets["motion"]
        )

        total = (
            self.speed_weight * speed_loss
            +
            self.position_weight * position_loss
            +
            self.yaw_weight * yaw_loss
            +
            self.motion_weight * motion_loss
        )

        if not torch.isfinite(total):

            raise RuntimeError(
                "Non-finite multitask loss."
            )

        return {
            "total": total,
            "speed": speed_loss.detach(),
            "position": position_loss.detach(),
            "yaw_rate": yaw_loss.detach(),
            "motion": motion_loss.detach()
        }


if __name__ == "__main__":

    criterion = MultiTaskLoss()

    outputs = {
        "speed": torch.randn(8),
        "position": torch.randn(8, 2),
        "yaw_rate": torch.randn(8),
        "motion_logits": torch.randn(8, 3)
    }

    targets = {
        "speed": torch.randn(8),
        "position": torch.randn(8, 2),
        "yaw_rate": torch.randn(8),
        "motion": torch.randint(
            0,
            3,
            (8,)
        )
    }

    losses = criterion(
        outputs,
        targets
    )

    print(
        "Loss:",
        losses["total"].item()
    )

    print(
        "LOSS TEST PASSED"
    )