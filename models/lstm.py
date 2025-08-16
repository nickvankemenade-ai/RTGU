import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 num_layers=2, num_classes=2, is_lm=False):
        super().__init__()
        self.is_lm = is_lm
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        if is_lm:
            self.fc = nn.Linear(hidden_dim, vocab_size)
        else:
            self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, hidden)

        if self.is_lm:
            return self.fc(out), (hn, cn)
        else:
            return self.fc(out[:, -1, :])
