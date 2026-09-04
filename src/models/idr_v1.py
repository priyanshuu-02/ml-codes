"""
IDR-V1 motion model.

Predicts vehicle motion from a 2 s window of smartphone IMU, conditioned on the last
trusted speed. Deliberately small: the previous model carried 985,195 parameters and its
trajectory error was dominated by a broken target rather than by capacity.

Differences from V8 that matter:

  * An acceleration head. The SIH statement asks for speed AND acceleration; V8 predicted
    only speed. The vehicle logs carry longitudinal acceleration directly, so it is a
    supervised target rather than a derivative of a prediction.
  * Log-variance heads that are actually trained. V8 exported three of them but its loss
    was plain SmoothL1, so those outputs were never supervised and shipped as noise. The
    Android runtime now reads them, so they are trained here with Gaussian NLL.
  * Outputs are returned as a fixed-order tuple so ONNX export names stay stable and the
    runtime can look them up by name.
"""
import torch
import torch.nn as nn


class IdrV1Model(nn.Module):
    """
    Conv1d feature extractor, GRU over time, then multi-task heads.

    Convolutions capture the short vibration and jerk patterns that carry speed
    information; the GRU integrates them across the window. Speed conditioning enters after
    the temporal encoder so it informs the heads without letting the network ignore the IMU
    entirely, which is the failure mode that made V8 look acceptable per window while
    integrating badly.
    """

    def __init__(
        self,
        input_channels=6,
        conv_dim=64,
        hidden_dim=96,
        speed_embedding=16,
        dropout=0.1,
        motion_classes=3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
        )

        self.temporal = nn.GRU(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.speed_encoder = nn.Sequential(
            nn.Linear(1, speed_embedding),
            nn.GELU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + speed_embedding, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Each regression head emits a value and its log variance.
        self.speed_head = nn.Linear(hidden_dim, 2)
        self.acceleration_head = nn.Linear(hidden_dim, 2)
        self.position_head = nn.Linear(hidden_dim, 4)
        self.heading_head = nn.Linear(hidden_dim, 2)
        self.motion_head = nn.Linear(hidden_dim, motion_classes)

        self._initialise()

    def _initialise(self):
        """Start every log variance near zero so early NLL gradients stay well behaved."""
        for head in (self.speed_head, self.acceleration_head, self.position_head, self.heading_head):
            nn.init.zeros_(head.bias)
            nn.init.xavier_uniform_(head.weight, gain=0.1)

    def forward(self, imu, initial_speed_normalized):
        """
        imu: (batch, window, channels)
        initial_speed_normalized: (batch,)

        Returns a fixed-order tuple:
            speed, speed_log_variance,
            acceleration, acceleration_log_variance,
            position, position_log_variance,
            heading_delta, heading_delta_log_variance,
            motion_logits
        """
        encoded = self.features(imu.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.temporal(encoded)
        # Last timestep summarises the window, which is what the endpoint targets describe.
        summary = sequence[:, -1, :]

        speed_context = self.speed_encoder(initial_speed_normalized.reshape(-1, 1))
        shared = self.shared(torch.cat([summary, speed_context], dim=1))

        speed_output = self.speed_head(shared)
        acceleration_output = self.acceleration_head(shared)
        position_output = self.position_head(shared)
        heading_output = self.heading_head(shared)
        motion_logits = self.motion_head(shared)

        return (
            speed_output[:, 0],
            speed_output[:, 1],
            acceleration_output[:, 0],
            acceleration_output[:, 1],
            position_output[:, :2],
            position_output[:, 2:],
            heading_output[:, 0],
            heading_output[:, 1],
            motion_logits,
        )

    @staticmethod
    def output_names():
        return [
            "speed",
            "speed_log_variance",
            "acceleration",
            "acceleration_log_variance",
            "position",
            "position_log_variance",
            "heading_delta",
            "heading_delta_log_variance",
            "motion_logits",
        ]

    @staticmethod
    def input_names():
        return ["imu", "initial_speed_normalized"]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IdrV1Loss(nn.Module):
    """
    Heteroscedastic multi-task loss.

    Each regression term is a Gaussian negative log likelihood,

        0.5 * (log_var + (error^2) * exp(-log_var))

    which supervises the log-variance heads instead of leaving them untrained as the
    previous loss did. It also lets the network down-weight windows it genuinely cannot
    predict rather than being forced to guess confidently.

    Log variances are clamped so a head cannot escape to infinite confidence, which would
    otherwise let it zero out its own gradient early in training.
    """

    def __init__(
        self,
        class_weights=(1.5, 1.0, 1.5),
        turn_multiplier=2.0,
        speed_weight=1.0,
        acceleration_weight=0.5,
        position_weight=2.0,
        forward_emphasis=3.0,
        heading_weight=4.0,
        motion_weight=0.75,
        # Tight bounds on purpose. With a wide range the network minimises the loss by
        # declaring near-infinite confidence on training samples, which drives the total
        # arbitrarily negative and destroys generalisation: an early run reached -25.5 in
        # training while validation climbed to +12.7. Bounding the variance keeps the term
        # a calibration signal rather than a free parameter to exploit.
        log_variance_bounds=(-2.0, 2.0),
    ):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.turn_multiplier = turn_multiplier
        self.speed_weight = speed_weight
        self.acceleration_weight = acceleration_weight
        self.position_weight = position_weight
        self.forward_emphasis = forward_emphasis
        self.heading_weight = heading_weight
        self.motion_weight = motion_weight
        self.low, self.high = log_variance_bounds

    def _nll(self, prediction, target, log_variance):
        log_variance = log_variance.clamp(self.low, self.high)
        squared_error = (prediction - target) ** 2
        return 0.5 * (log_variance + squared_error * torch.exp(-log_variance))

    def forward(self, outputs, targets):
        (speed, speed_log_variance,
         acceleration, acceleration_log_variance,
         position, position_log_variance,
         heading_delta, heading_log_variance,
         motion_logits) = outputs

        # Turning windows are the ones that decide trajectory accuracy, so they weigh more.
        turn_weight = torch.where(targets["motion"] == 2, self.turn_multiplier, 1.0)

        speed_term = self._nll(speed, targets["speed"], speed_log_variance)
        acceleration_term = self._nll(acceleration, targets["acceleration"], acceleration_log_variance)

        forward_term = self._nll(position[:, 0], targets["position"][:, 0], position_log_variance[:, 0])
        lateral_term = self._nll(position[:, 1], targets["position"][:, 1], position_log_variance[:, 1])
        position_term = self.forward_emphasis * forward_term + lateral_term

        heading_term = self._nll(heading_delta, targets["heading_delta"], heading_log_variance)

        motion_term = nn.functional.cross_entropy(
            motion_logits, targets["motion"], weight=self.class_weights, reduction="none"
        )

        total = (
            self.speed_weight * speed_term
            + self.acceleration_weight * acceleration_term
            + self.position_weight * position_term
            + self.heading_weight * heading_term
            + self.motion_weight * motion_term
        ) * turn_weight

        if not torch.isfinite(total).all():
            raise RuntimeError("Non-finite IDR-V1 loss.")

        return {
            "total": total.mean(),
            "speed": speed_term.mean().detach(),
            "acceleration": acceleration_term.mean().detach(),
            "forward": forward_term.mean().detach(),
            "lateral": lateral_term.mean().detach(),
            "heading_delta": heading_term.mean().detach(),
            "motion": motion_term.mean().detach(),
        }


if __name__ == "__main__":
    model = IdrV1Model()
    print(f"parameters: {model.parameter_count():,}")
    imu = torch.randn(8, 20, 6)
    speed = torch.randn(8)
    outputs = model(imu, speed)
    for name, value in zip(model.output_names(), outputs):
        print(f"  {name:<28} {tuple(value.shape)}")
