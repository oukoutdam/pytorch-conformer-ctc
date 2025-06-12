import torch
from torch import nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation for speech spectrograms.
    Applies frequency and time masking to improve model generalization.

    Reference: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """

    def __init__(
        self,
        freq_masks: int,
        freq_width: int,
        time_masks: int,
        time_width_ratio: float
    ) -> None:
        """
        Initialize SpecAugment parameters.

        Args:
            freq_masks: Number of frequency masks to apply
            freq_width: Maximum width of frequency masks
            time_masks: Number of time masks to apply
            time_width_ratio: Time mask width as ratio of sequence length
        """
        super().__init__()

        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width_ratio = time_width_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input spectrogram.

        Args:
            x: Input spectrogram (batch_size, n_mels, time_steps)

        Returns:
            Augmented spectrogram with same shape as input
        """
        # Calculate time mask width based on sequence length
        time_dim = x.size(-1)
        time_width = int(time_dim * self.time_width_ratio)

        # Apply frequency masks
        for _ in range(self.freq_masks):
            freq_masking = FrequencyMasking(freq_mask_param=self.freq_width)
            x = freq_masking(x)

        # Apply time masks
        for _ in range(self.time_masks):
            time_masking = TimeMasking(time_mask_param=time_width)
            x = time_masking(x)

        return x
