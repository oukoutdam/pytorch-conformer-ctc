import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Conformer
from .modules.conv_subsampling import StridingConvSubsampling
from .modules.preprocessor import AudioPreprocessor


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

        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_fft=512,
            n_mels=80,
            freq_masks=2,
            time_masks=10,
            freq_width=27,
            time_width=0.05
        )

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
        signals: torch.Tensor,
        signal_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            signals: Input audio signal (batch, time)
            signallens: Length of each audio in the batch

        Returns:
            logp: Log probabilities (batch, time, vocab_size)
            new_feat_lens: Updated sequence lengths after subsampling
        """
        # put through preprocessor
        feats, feat_lens = self.preprocessor(signals, signal_lens)

        # Apply subsampling
        x, new_feat_lens = self.subsampling(feats, feat_lens)

        # Pass through conformer encoder
        x, new_feat_lens = self.conformer_encoder(x, new_feat_lens)

        # Project to vocabulary size
        logits = self.fc(x)  # (batch, time, vocab)

        # Apply log softmax for CTC loss
        logp = F.log_softmax(logits, dim=-1)

        return logp, new_feat_lens
