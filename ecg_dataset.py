import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class ECGDataset(Dataset):
    def __init__(self, split="train"):
        # ECG5000 from HuggingFace datasets
        self.dataset = load_dataset("ecg5000")[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # "signal" is a list of floats, "label" is integer class (1–5)
        signal = torch.tensor(item["signal"], dtype=torch.float32)
        label = torch.tensor(item["label"] - 1, dtype=torch.long)  # shift to 0–4
        return signal, label
