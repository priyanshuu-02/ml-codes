import torch
import torch.nn as nn


class PatchTSTBranch(nn.Module):
    """
    Lightweight PatchTST-style temporal Transformer.

    Input:
        [B, T, C]

    Output:
        [B, T, hidden_dim]
    """

    def __init__(
        self,
        input_dim=96,
        hidden_dim=128,
        patch_len=4,
        stride=2,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        self.patch_projection = nn.Linear(
            input_dim * patch_len,
            hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x: [B, T, C]
        batch_size, sequence_length, channels = x.shape

        patches = []

        # Create temporal patches
        for start in range(
            0,
            sequence_length - self.patch_len + 1,
            self.stride
        ):

            patch = x[
                :,
                start:start + self.patch_len,
                :
            ]

            patch = patch.reshape(
                batch_size,
                -1
            )

            patches.append(patch)

        # [B, number_of_patches, patch_len*C]
        patches = torch.stack(
            patches,
            dim=1
        )

        # Patch embedding
        patches = self.patch_projection(
            patches
        )

        # Transformer
        features = self.transformer(
            patches
        )

        features = self.norm(
            features
        )

        features = self.output_projection(
            features
        )

        features = self.dropout(
            features
        )

        # ----------------------------------------------------
        # Interpolate patch-level representation back to
        # the original temporal resolution.
        # ----------------------------------------------------

        features = features.transpose(1, 2)

        features = torch.nn.functional.interpolate(
            features,
            size=sequence_length,
            mode="linear",
            align_corners=False
        )

        features = features.transpose(1, 2)

        return features


if __name__ == "__main__":

    model = PatchTSTBranch(
        input_dim=96,
        hidden_dim=128,
        patch_len=4,
        stride=2,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )

    x = torch.randn(
        128,
        20,
        96
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: {parameters:,}"
    )