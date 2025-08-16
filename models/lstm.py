import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, num_classes=2, num_layers=1, vocab_size=30522, is_ecg=False):
        super().__init__()
        self.is_ecg = is_ecg
        if not is_ecg:  # IMDB
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            input_size = embed_dim
        else:  # ECG
            input_size = 1
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask=None):
        if not self.is_ecg:  # IMDB
            x = self.embedding(x)
        else:  # ECG: add channel dimension
            x = x.unsqueeze(-1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
