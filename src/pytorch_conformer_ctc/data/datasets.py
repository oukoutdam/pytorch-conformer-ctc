from typing import Callable, Union
import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import json
from .transform import TextTransform


class CEJCDataset(Dataset):
    """
    Dataset class for CEJC (Common English-Japanese Corpus) speech recognition data.
    Loads audio files and text transcriptions from a manifest file.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        tokenizer: TextTransform,
        transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """
        Initialize the dataset.

        Args:
            manifest_path: Path to JSON manifest file containing audio/text pairs
            tokenizer: Text tokenizer for converting text to integer sequences
            transform: Audio feature extraction transform (e.g., MelSpectrogram)
        """
        # Load manifest file - each line is a JSON object with audio_filepath and text
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            feats: Audio features (time, feature_dim)
            labels: Text labels as integer sequence (sequence_length,)
        """
        # Extract text and audio path from manifest
        sample = self.samples[idx]
        text = sample["text"]
        audio_filepath = sample["audio_filepath"]

        # Load audio waveform
        waveform, _ = torchaudio.load(audio_filepath)

        # Apply feature transform and reshape
        # Input: [1, F, T] -> Output: [T, F] (time-first for RNN compatibility)
        feats = self.transform(waveform).squeeze(0).transpose(0, 1)

        # Convert text to integer tokens
        labels = torch.tensor(self.tokenizer.text_to_int(text), dtype=torch.long)

        return feats, labels


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    Pads sequences to the same length within each batch.

    Args:
        batch: List of (features, labels) tuples

    Returns:
        feats: Padded feature tensors (batch, max_time, feature_dim)
        flens: Feature sequence lengths (batch,)
        labs: Padded label tensors (batch, max_label_length)
        llens: Label sequence lengths (batch,)
    """
    # Separate features and labels
    feats, labels = zip(*batch)

    # Conver tuples to lists for pad_sequence()
    feats = list(feats)
    labels = list(labels)

    # Calculate sequence lengths before padding
    flens = torch.tensor([f.size(0) for f in feats])  # Feature lengths
    llens = torch.tensor([label.size(0) for label in labels])  # Label lengths

    # Pad sequences to same length within batch
    feats = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    labs = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=TextTransform.blank
    )

    return feats, flens, labs, llens
