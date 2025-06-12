# Text related transform for labels
class TextTransform:
    """
    Maps characters to integers and vice versa for text tokenization.
    Used for converting text to integer sequences for CTC training.
    """

    blank: int = 0  # CTC blank token is always index 0

    def __init__(self, tokens: list[str]) -> None:
        """
        Initialize the text transform with a vocabulary.

        Args:
            tokens: list of characters/tokens in vocabulary order
                   (blank token should be at index 0)
        """
        self.length = len(tokens)

        # Create bidirectional mappings
        self.char_map: dict[str, int] = {}    # character -> index
        self.index_map: dict[int, str] = {}   # index -> character

        for index, ch in enumerate(tokens):
            self.char_map[ch] = index
            self.index_map[index] = ch

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.length

    def text_to_int(self, text: str) -> list[int]:
        """
        Convert text to integer sequence using character mapping.

        Args:
            text: Input text string

        Returns:
            Integer sequence representing the text
        """
        int_sequence = []
        for c in text:
            if c in self.char_map:
                int_sequence.append(self.char_map[c])
            else:
                # Handle unknown characters with <unk> token
                int_sequence.append(self.char_map["<unk>"])
        return int_sequence

    def int_to_text(self, labels: list[int]) -> str:
        """
        Convert integer labels back to text sequence.

        Args:
            labels: Integer sequence or list of sequences

        Returns:
            Decoded text string
        """
        string = []
        for i in labels:
            string.append(self.index_map[i])

        # Join characters without spaces (direct character concatenation)
        return "".join(string)
