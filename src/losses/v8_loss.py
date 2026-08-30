import torch
import torch.nn as nn
import torch.nn.functional as F


class V8Loss(nn.Module):
    """V7 loss with a rollout-compatible integrated heading-change target."""
    def __init__(self, class_weights, turn_multiplier=2.0):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.turn_multiplier = turn_multiplier
        self.huber = nn.SmoothL1Loss(beta=1.0, reduction="none")
        self.heading_huber = nn.SmoothL1Loss(beta=0.10, reduction="none")

    def forward(self, outputs, targets):
        turn_weight = torch.where(targets["motion"] == 2, self.turn_multiplier, 1.0)
        speed = self.huber(outputs["speed"], targets["speed"])
        position_raw = self.huber(outputs["position"], targets["position"])
        position = 3.0 * position_raw[:, 0] + position_raw[:, 1]
        heading = self.heading_huber(outputs["heading_delta"], targets["heading_delta"])
        motion = F.cross_entropy(outputs["motion_logits"], targets["motion"], weight=self.class_weights, reduction="none")
        total = (speed + 2.0 * position + 4.0 * heading + 0.75 * motion) * turn_weight
        if not torch.isfinite(total).all(): raise RuntimeError("Non-finite V8 loss.")
        return {"total": total.mean(), "speed": speed.mean().detach(), "position": position.mean().detach(),
                "heading_delta": heading.mean().detach(), "motion": motion.mean().detach()}
