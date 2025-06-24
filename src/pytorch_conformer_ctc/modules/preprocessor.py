import torch
import torchaudio
from torch import nn
from typing import Union

class _SpecAugment(nn.Module):
    """Internal SpecAugment module."""
    def __init__(
        self,
        freq_masks: int,
        time_masks: int,
        freq_width: int,
        time_width: Union[int, float],
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_width)
        self.time_mask_param = time_width

    def forward(self, specgram: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return specgram

        specgram = specgram.clone()

        for _ in range(self.freq_masks):
            specgram = self.freq_masking(specgram)

        if self.time_masks > 0:
            for _ in range(self.time_masks):
                if isinstance(self.time_mask_param, float):
                    max_widths = (lengths.float() * self.time_mask_param).long()
                else:
                    max_widths = torch.full_like(lengths, self.time_mask_param)

                widths = (torch.rand(len(lengths), device=specgram.device) * max_widths).long()
                max_starts = (lengths - widths).clamp(min=0)
                starts = (torch.rand(len(lengths), device=specgram.device) * max_starts).long()

                time_dim_size = specgram.shape[2]
                time_range = torch.arange(time_dim_size, device=specgram.device)[None, :]
                mask = (time_range >= starts[:, None]) & (time_range < (starts + widths)[:, None])

                specgram = specgram.masked_fill(mask[:, None, :], 0.0)

        return specgram

# A helper function to create the mask for normalization, similar to what NeMo uses.
def make_seq_mask_like(lengths: torch.Tensor, like: torch.Tensor, time_dim: int, valid_ones: bool = True):
    """
    Create a sequence mask.

    Args:
        lengths: A tensor of sequence lengths (Batch,).
        like: A tensor to match the shape of (Batch, Dim, Time).
        time_dim: The time dimension index.
        valid_ones: If True, valid positions are 1s and padding is 0s. Otherwise, vice-versa.

    Returns:
        A boolean mask of shape (Batch, 1, Time) or (Batch, Time, 1) etc.
    """
    # Get the maximum length from the 'like' tensor
    max_len = like.shape[time_dim]

    # Create a range tensor [0, 1, 2, ..., max_len - 1]
    seq_range = torch.arange(max_len, device=lengths.device)

    # Expand lengths and seq_range for broadcasting
    # lengths: (B) -> (B, 1)
    # seq_range: (T) -> (1, T)
    # Result of comparison: (B, T)
    mask = seq_range[None, :] < lengths[:, None]

    # Reshape mask to match the 'like' tensor's dimensions for broadcasting
    # e.g., if like is (B, D, T) and time_dim is 2, mask becomes (B, 1, T)
    mask_shape = [1] * like.dim()
    mask_shape[0] = like.shape[0]
    mask_shape[time_dim] = max_len
    mask = mask.view(*mask_shape)

    if not valid_ones:
        mask = ~mask

    return mask

class AudioPreprocessor(nn.Module):
    """
    An all-in-one module to convert raw audio to normalized and augmented
    log-Mel spectrograms.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        n_mels: int = 80,
        n_fft: int = 512,
        dither: float = 1e-5,
        # SpecAugment specific parameters
        apply_spec_augment: bool = True,
        freq_masks: int = 2,
        time_masks: int = 2,
        freq_width: int = 27,
        time_width: Union[int, float] = 100,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.dither = dither
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)
        self.hop_length = hop_length
        self.eps = 1e-9

        self.mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels, window_fn=torch.hann_window,
            power=2.0, center=True, pad_mode="reflect",
        )

        # Instantiate SpecAugment internally
        self.apply_spec_augment = apply_spec_augment
        if apply_spec_augment:
            self.spec_augment = _SpecAugment(
                freq_masks=freq_masks, time_masks=time_masks,
                freq_width=freq_width, time_width=time_width
            )

    def _apply_dithering(self, signals: torch.Tensor) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            return signals + torch.randn_like(signals) * self.dither
        return signals

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(input_lengths, self.hop_length, rounding_mode="floor").add(1).long()

    def _apply_normalization(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = make_seq_mask_like(lengths=lengths, like=features, time_dim=-1, valid_ones=False)
        features = features.masked_fill(mask, 0.0)
        lengths_for_div = lengths.view(-1, 1, 1).float()
        mean = features.sum(dim=-1, keepdim=True) / lengths_for_div
        variance = (features.pow(2).sum(dim=-1, keepdim=True) / lengths_for_div) - mean.pow(2)
        variance = variance * lengths_for_div / (lengths_for_div - 1.0).clamp(min=1.0)
        std = torch.sqrt(variance.clamp(min=self.eps))
        features = (features - mean) / (std + self.eps)
        features = features.masked_fill(mask, 0.0)
        return features

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes raw audio into augmented, normalized features."""

        # 1. Dithering
        signals = self._apply_dithering(input_signal)

        # 2. Mel Spectrogram -> (Batch, Freq, Time)
        features = self.mel_spec_extractor(signals)

        # 3. Log conversion
        features = features.clamp(min=self.eps).log()

        # 4. Compute feature lengths
        feature_lengths = self._compute_output_lengths(length)

        # 5. Per-feature normalization
        features = self._apply_normalization(features, feature_lengths)

        # 6. Apply SpecAugment (if enabled and in training mode)
        # This happens on the (Batch, Freq, Time) tensor.
        if self.apply_spec_augment and self.training:
            features = self.spec_augment(features, feature_lengths)

        # 7. Final transpose for model input -> (Batch, Time, Freq)
        features = features.transpose(1, 2)

        return features, feature_lengths
