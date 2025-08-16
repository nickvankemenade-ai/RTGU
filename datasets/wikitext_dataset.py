import re
from collections import Counter
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class WikiTextDataset(Dataset):
    """
    Word-level LM dataset for WikiText-2 (raw). Produces fixed-length blocks.

    Each sample is a LongTensor of shape (block_size + 1,):
      x = sample[:-1], y = sample[1:]
    """
    def __init__(self, split="train", block_size=128, vocab=None, min_freq=2, max_vocab_size=50000):
        super().__init__()
        # Map common alias
        split = "validation" if split in ("valid", "validation") else split
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")

        raw = load_dataset("wikitext", "wikitext-2-raw-v1")[split]
        text = "\n".join(raw["text"])

        # Simple tokenization: whitespace with basic punctuation splitting
        tokens = self._tokenize(text)

        # Build or reuse vocab (train builds; val/test must reuse train vocab)
        if vocab is None:
            if split != "train":
                raise ValueError("Pass the training vocab to validation/test datasets.")
            self.vocab, self.itos = self._build_vocab(tokens, min_freq=min_freq, max_size=max_vocab_size)
        else:
            self.vocab = vocab
            self.itos = [None] * len(vocab)  # not needed for training

        # Encode tokens with UNK fallback
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

        # Create contiguous blocks of length block_size+1
        T = block_size + 1
        n = len(ids) // T
        ids = ids[: n * T]
        self.data = torch.tensor(ids, dtype=torch.long).view(n, T)

    @staticmethod
    def _tokenize(text):
        # Split punctuation, keep contractions; very lightweight
        text = re.sub(r"([.,!?;:\"()\[\]{}])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split(" ")

    @staticmethod
    def _build_vocab(tokens, min_freq=2, max_size=50000):
        counter = Counter(tokens)
        # Special tokens
        itos = ["<unk>", "<pad>"]
        # most common tokens above min_freq
        common = [tok for tok, c in counter.most_common() if c >= min_freq]
        if max_size is not None:
            common = common[: max(0, max_size - len(itos))]
        itos.extend(common)
        vocab = {tok: i for i, tok in enumerate(itos)}
        return vocab, itos

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        # Returns a (block_size+1,) tensor of token ids
        return self.data[idx]
