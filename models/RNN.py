import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, num_classes=5, cuda=True, device='cuda'):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, notes):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        nn.init.xavier_normal_(h)
        nn.init.xavier_normal_(c)
        h = h.to(self.device)
        c = c.to(self.device)
        x = x.to(self.device)

        output, _ = self.lstm(x, (h, c))

        out = self.fc2(self.relu(self.fc1(output[:, -1, :])))

        return out


class MMRNN(nn.ModuleList):
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, num_classes=5, embed_size=768, device="cuda"):
        super(MMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, embed_size)
        self.fc2 = nn.Linear(embed_size, num_classes)

        self.lnorm_out = nn.LayerNorm(embed_size)
        self.lnorm_embed = nn.LayerNorm(embed_size)

    def forward(self, x, note):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        nn.init.xavier_normal_(h)
        nn.init.xavier_normal_(c)
        h = h.to(self.device)
        c = c.to(self.device)
        x = x.to(self.device)
        note = note.to(self.device)

        output, _ = self.lstm(x, (h, c))
        # Take last hidden state
        out = self.fc1(output[:, -1, :])

        note = self.lnorm_embed(note)
        out = self.lnorm_out(out)
        out = note + out

        out = self.fc2(out)

        return out.squeeze(1)
