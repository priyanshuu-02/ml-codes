"""V8 changes the orientation target from endpoint yaw-rate to heading delta."""
from src.models.hybrid_v7 import V7DeadReckoningModel


class V8DeadReckoningModel(V7DeadReckoningModel):
    def forward(self, x, initial_speed):
        outputs = super().forward(x, initial_speed)
        outputs["heading_delta"] = outputs.pop("yaw_rate")
        outputs["heading_delta_log_variance"] = outputs.pop("yaw_rate_log_variance")
        return outputs
