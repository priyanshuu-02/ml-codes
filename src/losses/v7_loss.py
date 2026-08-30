import torch
import torch.nn as nn
import torch.nn.functional as F


class V7Loss(nn.Module):
    """Prioritizes longitudinal/turn behaviour while retaining all V1 tasks."""
    def __init__(self, class_weights, turn_multiplier=2.0, forward_weight=3.0, lateral_weight=1.0):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.turn_multiplier = turn_multiplier
        self.forward_weight, self.lateral_weight = forward_weight, lateral_weight
        self.huber = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def forward(self, outputs, targets):
        turn_weight = torch.where(targets["motion"] == 2, self.turn_multiplier, 1.0)
        speed = self.huber(outputs["speed"], targets["speed"])
        yaw = self.huber(outputs["yaw_rate"], targets["yaw_rate"])
        position_raw = self.huber(outputs["position"], targets["position"])
        position = (self.forward_weight * position_raw[:, 0] + self.lateral_weight * position_raw[:, 1])
        motion = F.cross_entropy(outputs["motion_logits"], targets["motion"], weight=self.class_weights, reduction="none")
        total = (speed + 2.0 * position + yaw + 0.75 * motion) * turn_weight
        if not torch.isfinite(total).all(): raise RuntimeError("Non-finite V7 loss.")
        return {"total": total.mean(), "speed": speed.mean().detach(), "position": position.mean().detach(),
                "yaw_rate": yaw.mean().detach(), "motion": motion.mean().detach()}
