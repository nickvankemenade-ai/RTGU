import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

class IMDBDataset(Dataset):
    def __init__(self, split="train", max_len=256):
        self.dataset = load_dataset("imdb")[split]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return (
            tokens.input_ids.squeeze(0), 
            tokens.attention_mask.squeeze(0), 
            torch.tensor(item["label"], dtype=torch.long)
        )
