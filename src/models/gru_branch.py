import torch
import torch.nn as nn


class GRUBranch(nn.Module):
    """
    Causal GRU branch for temporal vehicle-motion modeling.

    Input:
        [B, T, C]

    Output:
        [B, T, hidden_dim]
    """

    def __init__(
        self,
        input_dim=96,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=False
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        output_dim = (
            hidden_dim * 2
            if bidirectional
            else hidden_dim
        )

        self.projection = nn.Linear(
            output_dim,
            hidden_dim
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x: [B, T, C]
        output, _ = self.gru(x)

        # [B, T, hidden_dim]
        output = self.projection(output)

        output = self.norm(output)

        output = self.dropout(output)

        return output


if __name__ == "__main__":

    model = GRUBranch(
        input_dim=96,
        hidden_dim=128,
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