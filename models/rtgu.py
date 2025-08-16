import torch
import torch.nn as nn

class RTGUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_r.weight)
        nn.init.orthogonal_(self.linear_z.weight)
        nn.init.orthogonal_(self.linear_h.weight)
        nn.init.zeros_(self.linear_r.bias)
        nn.init.constant_(self.linear_z.bias, -2.0)
        nn.init.zeros_(self.linear_h.bias)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.linear_r(combined))
        z = torch.sigmoid(self.linear_z(combined))
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.linear_h(combined_r))
        return (1 - z) * h + z * h_tilde


class RTGU(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 num_layers=2, num_classes=2, is_lm=False):
        super().__init__()
        self.is_lm = is_lm
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.cells = nn.ModuleList(
            [RTGUCell(embed_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_layers)]
        )

        if is_lm:
            self.fc = nn.Linear(hidden_dim, vocab_size)
        else:
            self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
             for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(xt, h[i])
                xt = h[i]
            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        if self.is_lm:
            return self.fc(outputs), h
        else:
            return self.fc(outputs[:, -1, :])
