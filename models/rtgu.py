import torch
import torch.nn as nn

class RTGUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Gates take [x, h] (concat)
        self.linear_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_z = nn.Linear(input_size + hidden_size, hidden_size)

        # Candidate takes [x, r*h]
        self.linear_h = nn.Linear(input_size + hidden_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_r.weight)
        nn.init.orthogonal_(self.linear_z.weight)
        nn.init.orthogonal_(self.linear_h.weight)
        nn.init.zeros_(self.linear_r.bias)
        nn.init.constant_(self.linear_z.bias, -2.0)  # encourage initial carry-over
        nn.init.zeros_(self.linear_h.bias)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.linear_r(combined))
        z = torch.sigmoid(self.linear_z(combined))

        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.linear_h(combined_r))

        return (1 - z) * h + z * h_tilde


class RTGU(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, num_layers=1,
                 num_classes=2, vocab_size=30522, is_ecg=False):
        super().__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.is_ecg = is_ecg

        # Input size differs: IMDB = embedding, ECG = raw 1D
        if not is_ecg:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            input_size = embed_dim
        else:
            input_size = 1  # each ECG sample is scalar at each timestep

        # Stack of RTGU cells
        self.cells = nn.ModuleList(
            [RTGUCell(input_size if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_layers)]
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attention_mask=None):
        if not self.is_ecg:
            x = self.embedding(x)           # (batch, seq, embed_dim)
        else:
            x = x.unsqueeze(-1)             # (batch, seq, 1)

        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        for t in range(seq_len):
            xt = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(xt, h[i])
                xt = h[i]

        return self.fc(h[-1])
