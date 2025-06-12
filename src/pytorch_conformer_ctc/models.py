import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Conformer
from .modules.conv_subsampling import StridingConvSubsampling


class PytorchConformerCTC(nn.Module):
    """Conformer-based CTC model for speech recognition."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        d_model: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float,
        subsampling_factor: int,
        ff_expansion_factor: int,
        vocab_size: int,
    ) -> None:
        super().__init__()

        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Calculate feed-forward network dimension
        ffn_dim = d_model * ff_expansion_factor

        # Subsampling layer to reduce sequence length
        self.subsampling = StridingConvSubsampling(
            input_dim=input_dim,
            d_model=d_model,
            subsampling_factor=subsampling_factor
        )

        # Conformer encoder blocks
        self.conformer_encoder = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            dropout=dropout,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )

        # Final projection to vocabulary
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            feats: Input features (batch, time, feature_dim)
            feat_lens: Length of each sequence in the batch

        Returns:
            logp: Log probabilities (batch, time, vocab_size)
            new_feat_lens: Updated sequence lengths after subsampling
        """
        # Apply subsampling
        x, new_feat_lens = self.subsampling(feats, feat_lens)

        # Pass through conformer encoder
        x, new_feat_lens = self.conformer_encoder(x, new_feat_lens)

        # Project to vocabulary size
        logits = self.fc(x)  # (batch, time, vocab)

        # Apply log softmax for CTC loss
        logp = F.log_softmax(logits, dim=-1)

        return logp, new_feat_lens
