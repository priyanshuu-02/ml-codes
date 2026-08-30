import torch
import torch.nn as nn


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # [B, C, T] -> [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class ConvNeXt1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        expansion=4,
        kernel_size=7,
        dropout=0.1
    ):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim
        )

        self.norm = LayerNorm1d(dim)

        hidden_dim = dim * expansion

        self.pw1 = nn.Conv1d(
            dim,
            hidden_dim,
            kernel_size=1
        )

        self.activation = nn.GELU()

        self.pw2 = nn.Conv1d(
            hidden_dim,
            dim,
            kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

        self.gamma = nn.Parameter(
            1e-6 * torch.ones(dim)
        )

    def forward(self, x):

        residual = x

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.activation(x)
        x = self.pw2(x)
        x = self.dropout(x)

        # Layer scale
        x = x * self.gamma.view(1, -1, 1)

        return residual + x


class ConvNeXt1D(nn.Module):
    """
    Lightweight ConvNeXt-1D feature extractor.

    Input:
        [B, T, C]

    Output:
        [B, T, embed_dim]
    """

    def __init__(
        self,
        input_channels=6,
        embed_dim=96,
        depth=3,
        kernel_size=7,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            input_channels,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        self.blocks = nn.Sequential(
            *[
                ConvNeXt1DBlock(
                    dim=embed_dim,
                    expansion=4,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = LayerNorm1d(
            embed_dim
        )

    def forward(self, x):

        # Input [B, T, C]
        x = x.transpose(1, 2)

        # [B, C, T]
        x = self.input_projection(x)

        x = self.blocks(x)

        x = self.final_norm(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":

    model = ConvNeXt1D(
        input_channels=6,
        embed_dim=96,
        depth=3,
        dropout=0.1
    )

    x = torch.randn(
        128,
        20,
        6
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