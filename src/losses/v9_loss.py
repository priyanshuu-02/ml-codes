import torch
import torch.nn as nn

from src.losses.v8_loss import V8Loss


class V9Loss(nn.Module):
    """Local multitask loss plus global trajectory loss across an outage sequence."""
    def __init__(self, class_weights, position_mean, position_std, trajectory_weight=0.30):
        super().__init__()
        self.local = V8Loss(class_weights)
        self.register_buffer("position_mean", torch.as_tensor(position_mean, dtype=torch.float32))
        self.register_buffer("position_std", torch.as_tensor(position_std, dtype=torch.float32))
        self.trajectory_weight = trajectory_weight
        self.trajectory_huber = nn.SmoothL1Loss(beta=5.0)

    def forward(self, sequence_outputs, targets):
        steps = len(sequence_outputs)
        local_total = 0.0
        predicted_position, predicted_heading = [], []
        for step, output in enumerate(sequence_outputs):
            step_targets = {key: targets[key][:, step] for key in ("speed", "position", "heading_delta", "motion")}
            local_total = local_total + self.local(output, step_targets)["total"]
            predicted_position.append(output["position"] * self.position_std + self.position_mean)
            predicted_heading.append(output["heading_delta"])
        local_total = local_total / steps
        pred_pos = torch.stack(predicted_position, dim=1)
        pred_heading = torch.stack(predicted_heading, dim=1)
        true_pos = targets["position"] * self.position_std + self.position_mean
        zero = torch.zeros_like(pred_heading[:, :1])
        pred_theta = torch.cumsum(torch.cat([zero, pred_heading[:, :-1]], dim=1), dim=1)
        true_theta = torch.cumsum(torch.cat([zero, targets["heading_delta"][:, :-1]], dim=1), dim=1)
        def rotate(pos, theta):
            forward, lateral = pos[..., 0], pos[..., 1]
            return torch.stack([forward * torch.sin(theta) + lateral * torch.cos(theta),
                                forward * torch.cos(theta) - lateral * torch.sin(theta)], dim=-1)
        pred_global = torch.cumsum(rotate(pred_pos, pred_theta), dim=1)
        true_global = torch.cumsum(rotate(true_pos, true_theta), dim=1)
        trajectory = self.trajectory_huber(pred_global, true_global)
        total = local_total + self.trajectory_weight * trajectory
        if not torch.isfinite(total): raise RuntimeError("Non-finite V9 loss.")
        return {"total": total, "local": local_total.detach(), "trajectory": trajectory.detach()}
