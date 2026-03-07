"""
PyTorch model architectures for the NN vs PCR comparison.

Two architectures designed for small-sample spectral regression (299 features,
145 training samples):

1. SpectralCNN — 1D convolutional network that exploits wavelength locality.
                 Adjacent 2nd-derivative features are spatially correlated;
                 Conv1d kernels learn local spectral patterns (absorption peaks,
                 shoulders) with weight sharing across wavelength positions.
                 ~3,000 params.

2. TightPCAMLP — Minimal nonlinear extension of PCA regression. 15 PCA dims
                 (bottom PCs are noise), single hidden layer (16 units) with
                 BatchNorm. Tests whether the PC→pigment mapping has exploitable
                 nonlinearity. ~305 params.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required for the NN experiments. Install it with:\n"
        "  pip install torch\n"
        "or visit https://pytorch.org/get-started/locally/ for platform-specific instructions."
    )


class SpectralCNN(nn.Module):
    """
    1D convolutional network for spectral regression.

    Architecture:
        Input (batch, 299) → unsqueeze → (batch, 1, 299)
        Conv1d(1→16, k=7, pad=3) + BN + ReLU → AvgPool(4)   → (batch, 16, 74)
        Conv1d(16→32, k=5, pad=2) + BN + ReLU → AdaptiveAvgPool(8) → (batch, 32, 8)
        Flatten → Dropout → Linear(256→1)

    Why convolutions work here:
    - Pigment absorption features are 10-30 nm wide in the 2nd derivative
    - The 1st conv layer has a 7-point receptive field = 7 nm window
    - After 4× average pooling, the 2nd conv sees 5×4 = 20 nm of the
      original spectrum — a good match for absorption feature widths
    - Weight sharing across wavelength positions is a strong inductive bias:
      the same "absorption peak detector" can fire at any wavelength,
      dramatically reducing the parameter count vs a fully connected layer

    The key advantage over MLPs: convolutions enforce locality. An MLP
    treats feature 0 and feature 298 as equally related, but in a spectrum,
    nearby wavelengths are highly correlated and distant ones are not.
    Conv1d respects this spatial structure.

    BatchNorm normalizes activations per channel across the batch, which
    stabilizes training and allows higher learning rates. It's placed after
    the conv and before the activation (the "pre-activation" convention).
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim

        self.features = nn.Sequential(
            # Block 1: 7-wide kernels detect narrow spectral features
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),  # 299 → 74 (floor division)

            # Block 2: 5-wide kernels on the pooled representation combine
            # nearby features into broader spectral patterns
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # AdaptiveAvgPool1d(8) always outputs length 8 regardless of input
            # length — this makes the architecture robust to small changes in
            # input_dim (e.g., if we ever change the wavelength range)
            nn.AdaptiveAvgPool1d(8),  # 74 → 8
        )

        # 32 channels × 8 spatial positions = 256 features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32 * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 299) — flat spectral features
        # Conv1d expects (batch, channels, length), so we add a channel dim
        x = x.unsqueeze(1)  # (batch, 1, 299)
        x = self.features(x)  # (batch, 32, 8)
        x = x.flatten(1)  # (batch, 256)
        return self.head(x).squeeze(-1)  # (batch,)


class TightPCAMLP(nn.Module):
    """
    Minimal MLP on PCA-reduced features: pca_dim → hidden → BN → ReLU → Drop → 1.

    With pca_dim=15 and hidden_dim=16, this has:
      - Linear(15→16): 15×16 + 16 = 256 weights
      - BN(16): 16 + 16 = 32 params (scale + shift)
      - Linear(16→1): 16 + 1 = 17 weights
      - Total: ~305 params

    PCA reduction happens OUTSIDE this module (in the trainer), so this
    receives already-projected features. Compared to the old PCAMLP:
      - Fewer PCA dims (15 vs 30) — bottom PCs capture noise, not signal
      - Single hidden layer (vs two) — less capacity, less overfitting risk
      - BatchNorm (vs none) — stabilizes training on small batches
      - Lower dropout (0.2 vs 0.3) — fewer params need less regularization

    The question this tests: is there exploitable nonlinearity in the
    PC→pigment mapping, or is linear regression on PCs already optimal?
    """

    def __init__(self, pca_dim: int, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pca_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(arch_name: str, input_dim: int, config: dict) -> nn.Module:
    """
    Factory function to create a model by architecture name.

    Args:
        arch_name: One of "SpectralCNN", "TightPCAMLP".
        input_dim: Number of input features (299 for full spectra, 15 for PCA).
        config: The full CONFIG dict — pulls architecture params from config["nn"].

    Returns:
        An initialized PyTorch module.
    """
    nn_cfg = config["nn"]

    if arch_name == "SpectralCNN":
        return SpectralCNN(input_dim, dropout=nn_cfg["spectral_cnn_dropout"])
    elif arch_name == "TightPCAMLP":
        return TightPCAMLP(
            pca_dim=input_dim,
            hidden_dim=nn_cfg["tight_pcamlp_hidden_dim"],
            dropout=nn_cfg["tight_pcamlp_dropout"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
