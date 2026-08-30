"""V7: V1 encoder with a causal navigation-state conditioning branch."""
import torch
import torch.nn as nn

from src.models.hybrid import IntelligentDeadReckoningModel


class V7DeadReckoningModel(IntelligentDeadReckoningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 128)
        self.state_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.state_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, initial_speed):
        features = self.convnext(x)
        fused = self.fusion(self.gru(features), self.patchtst(features))
        state = self.state_encoder(initial_speed.reshape(-1, 1))
        return self.heads(self.state_norm(fused[:, -1, :] + state))
