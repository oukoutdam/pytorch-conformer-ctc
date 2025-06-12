import torch
from torch import nn
from typing import Tuple


class StridingConvSubsampling(nn.Module):
    """
    Strided convolution subsampling layer that reduces sequence length by a factor.
    Uses 2D convolutions with stride=2 to downsample the temporal dimension.
    """

    def __init__(self, input_dim: int, subsampling_factor: int, d_model: int) -> None:
        super().__init__()

        assert subsampling_factor in [2, 4, 8], "Subsampling factor must be either [2, 4, 8]"

        self.subsampling_factor = subsampling_factor
        conv_channels = d_model
        self.kernel_size = 3
        self.stride = 2

        # Calculate number of conv layers needed (log_2 of subsampling factor)
        self.num_layers = int(torch.log2(torch.tensor(subsampling_factor)).item())

        # Build conv layers - each layer reduces time dimension by factor of 2
        layers = []
        in_channels = 1  # Start with single channel input

        for _ in range(self.num_layers):
            layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2))
            layers.append(nn.ReLU())
            in_channels = conv_channels

        self.conv = nn.Sequential(*layers)

        # Linear projection to match conformer input dimension
        freq_dim_after_conv = calc_conv_out_dim_per_channel(
            input_dim, kernel_size=3, stride=2, num_layers=self.num_layers
        )
        linear_input_dim = conv_channels * freq_dim_after_conv
        self.linear = nn.Linear(linear_input_dim, d_model)

        # Precompute factors for efficient length calculation
        self._length_factor = subsampling_factor       # = 2**num_layers
        self._length_offset = subsampling_factor - 1   # = 2**num_layers - 1

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through subsampling layer.

        Args:
            x: Input features (batch, time, feature_dim)
            lengths: Sequence lengths for each batch element

        Returns:
            x: Subsampled features (batch, time', d_model)
            new_lengths: Updated sequence lengths after subsampling
        """
        # Add channel dimension: [B, T, F] -> [B, 1, T, F]
        x = x.unsqueeze(1)

        # Apply convolutions: [B, 1, T, F] -> [B, C, T', F']
        x = self.conv(x)

        # Reshape for linear layer
        b, c, t, f = x.size()
        x = x.transpose(1, 2)  # [B, C, T', F'] -> [B, T', C, F']
        x = x.contiguous().view(b, t, c * f)  # [B, T', C*F']

        # Project to model dimension
        x = self.linear(x)  # [B, T', d_model]

        # Calculate new sequence lengths efficiently
        new_lengths = (lengths - self._length_offset) // self._length_factor
        new_lengths = torch.clamp(new_lengths, min=1)

        # REFERENCE: Alternative length calculation methods
        # 1. Simple approximation (less accurate):
        # lengths = torch.div(lengths, self.subsampling_factor, rounding_mode="floor")
        # lengths = torch.clamp(lengths, min=1)

        # 2. Exact calculation (more computation):
        # new_lengths = lengths
        # for _ in range(self.num_layers):
        #     new_lengths = (new_lengths - self.kernel_size) // self.stride + 1

        # 3. Efficient and accurate (current method):
        # Uses precomputed factors for O(1) calculation

        return x, new_lengths


def calc_conv_out_dim_per_channel(
    input_dim: int,
    kernel_size: int,
    stride: int,
    num_layers: int
) -> int:
    """
    Calculate output dimension after applying multiple conv layers.

    Args:
        input_dim: Input feature dimension
        kernel_size: Convolution kernel size
        stride: Convolution stride
        num_layers: Number of conv layers to apply

    Returns:
        Final output dimension
    """
    dim = input_dim
    for _ in range(num_layers):
        # Apply conv formula: floor((dim - kernel_size)/stride + 1)
        dim = (dim - kernel_size) // stride + 1
    return dim
