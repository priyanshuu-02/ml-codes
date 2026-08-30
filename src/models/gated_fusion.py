import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Learnable gated fusion of GRU and PatchTST features.

    GRU features:
        [B, T, D]

    PatchTST features:
        [B, T, D]

    Output:
        [B, T, D]
    """

    def __init__(
        self,
        feature_dim=128,
        dropout=0.1
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(
                feature_dim * 2,
                feature_dim
            ),
            nn.GELU(),
            nn.Linear(
                feature_dim,
                feature_dim
            ),
            nn.Sigmoid()
        )

        self.output_projection = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.norm = nn.LayerNorm(
            feature_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, gru_features, transformer_features):

        if gru_features.shape != transformer_features.shape:
            raise ValueError(
                "GRU and Transformer feature shapes "
                "must match. "
                f"Got {gru_features.shape} and "
                f"{transformer_features.shape}"
            )

        combined = torch.cat(
            [
                gru_features,
                transformer_features
            ],
            dim=-1
        )

        gate = self.gate(
            combined
        )

        fused = (
            gate * gru_features
            +
            (1.0 - gate) * transformer_features
        )

        fused = self.output_projection(
            fused
        )

        fused = self.dropout(
            fused
        )

        fused = self.norm(
            fused
        )

        return fused


if __name__ == "__main__":

    model = GatedFusion(
        feature_dim=128,
        dropout=0.1
    )

    gru = torch.randn(
        128,
        20,
        128
    )

    transformer = torch.randn(
        128,
        20,
        128
    )

    output = model(
        gru,
        transformer
    )

    print("GRU input        :", gru.shape)
    print("Transformer input:", transformer.shape)
    print("Fused output     :", output.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: {parameters:,}"
    )