import torch
import torch.nn as nn

from src.models.convnext1d import ConvNeXt1D
from src.models.gru_branch import GRUBranch
from src.models.patchtst import PatchTSTBranch
from src.models.gated_fusion import GatedFusion
from src.models.heads import PredictionHeads


class IntelligentDeadReckoningModel(nn.Module):

    def __init__(
        self,
        input_channels=6,
        conv_dim=96,
        hidden_dim=128,
        dropout=0.1
    ):
        super().__init__()

        # ----------------------------------------------------
        # ConvNeXt-1D feature extractor
        # ----------------------------------------------------

        self.convnext = ConvNeXt1D(
            input_channels=input_channels,
            embed_dim=conv_dim,
            depth=3,
            dropout=dropout
        )

        # ----------------------------------------------------
        # GRU temporal branch
        # ----------------------------------------------------

        self.gru = GRUBranch(
            input_dim=conv_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # ----------------------------------------------------
        # PatchTST Transformer branch
        # ----------------------------------------------------

        self.patchtst = PatchTSTBranch(
            input_dim=conv_dim,
            hidden_dim=hidden_dim,
            patch_len=4,
            stride=2,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Learnable gated fusion
        # ----------------------------------------------------

        self.fusion = GatedFusion(
            feature_dim=hidden_dim,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Multi-task prediction heads
        # ----------------------------------------------------

        self.heads = PredictionHeads(
            input_dim=hidden_dim,
            hidden_dim=64,
            dropout=dropout,
            num_motion_classes=3
        )

    def forward(self, x):

        # Input:
        # [Batch, Time, Channels]
        #
        # Example:
        # [128, 20, 6]

        # ----------------------------------------------------
        # ConvNeXt feature extraction
        # ----------------------------------------------------

        features = self.convnext(x)

        # [B, T, 96]

        # ----------------------------------------------------
        # GRU branch
        # ----------------------------------------------------

        gru_features = self.gru(features)

        # [B, T, 128]

        # ----------------------------------------------------
        # PatchTST Transformer branch
        # ----------------------------------------------------

        transformer_features = self.patchtst(features)

        # [B, T, 128]

        # ----------------------------------------------------
        # Gated fusion
        # ----------------------------------------------------

        fused = self.fusion(
            gru_features,
            transformer_features
        )

        # [B, T, 128]

        # ----------------------------------------------------
        # Use latest timestep
        #
        # This keeps the final prediction causal for
        # real-time dead-reckoning inference.
        # ----------------------------------------------------

        final_features = fused[:, -1, :]

        # [B, 128]

        # ----------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------

        outputs = self.heads(
            final_features
        )

        return outputs


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("INTELLIGENT DEAD RECKONING MODEL TEST")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = IntelligentDeadReckoningModel(
        input_channels=6,
        conv_dim=96,
        hidden_dim=128,
        dropout=0.1
    ).to(device)

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    x = torch.randn(
        8,
        20,
        6,
        device=device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(x)

    # --------------------------------------------------------
    # Display shapes
    # --------------------------------------------------------

    print(
        f"\nInput shape: "
        f"{tuple(x.shape)}"
    )

    print("\nOutput shapes:")

    for name, value in outputs.items():

        print(
            f"  {name}: "
            f"{tuple(value.shape)}"
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{parameters:,}"
    )

    print(
        "\nMODEL TEST SUCCESSFUL"
    )