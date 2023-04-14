import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, ecg_channels=12):
        super(CNN, self).__init__()
        self.name = "CNN"
        self.conv1 = nn.Conv1d(ecg_channels, 16, 7)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(32, 48, 3)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.fc0 = nn.Linear(5856, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 5)
        self.activation = nn.ReLU()
    def forward(self, x, notes):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x


class MMCNN_SUM(nn.Module):
    def __init__(self, ecg_channels=12):
        super(MMCNN_SUM, self).__init__()
        # ECG processing Layers
        self.name = "MMCNN_SUM"
        self.conv1 = nn.Conv1d(ecg_channels, 16, 7)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(32, 48, 3)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.fc0 = nn.Linear(5856, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 5)

        # Clinical Notes Processing Layers
        self.fc_emb = nn.Linear(768, 128)
        self.norm = nn.LayerNorm(128)

        self.activation = nn.ReLU()

    def forward(self, x, notes):
        # ECG Processing
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))

        # Notes Processing
        notes = notes.view(notes.size(0),-1)
        notes = self.activation(self.fc_emb(notes))

        x = self.fc2(self.norm(x + notes)) 
        x = x.squeeze(1)
        return x

class MMCNN_CAT(nn.Module):
    def __init__(self, ecg_channels=12):
        super(MMCNN_CAT, self).__init__()
        # ECG processing Layers
        self.name = "MMCNN_CAT"
        self.conv1 = nn.Conv1d(ecg_channels, 16, 7)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(32, 48, 3)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.fc0 = nn.Linear(5856, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(256, 5)

        # Clinical Notes Processing Layers
        self.fc_emb = nn.Linear(768, 128)
        self.norm = nn.LayerNorm(128)

        self.activation = nn.ReLU()

    def forward(self, x, notes):
        # ECG Processing
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))

        # Notes Processing
        notes = notes.view(notes.size(0),-1)
        notes = self.activation(self.fc_emb(notes))

        x = self.fc2(torch.cat((x,notes),dim=1))
        x = x.squeeze(1)
        return x
class MMCNN_ATT(nn.Module):
    def __init__(self, ecg_channels=12):
        super(MMCNN_ATT, self).__init__()
        # ECG processing Layers
        self.name = "MMCNN_ATT"
        self.conv1 = nn.Conv1d(ecg_channels, 16, 7)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(32, 48, 3)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.fc0 = nn.Linear(5856, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 5)

        # Clinical Notes Processing Layers
        self.fc_emb = nn.Linear(768, 128)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)

        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.activation = nn.ReLU()

    def forward(self, x, notes):
        # ECG Processing
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.norm1(x)

        # Notes Processing
        notes = notes.view(notes.size(0),-1)
        notes = self.activation(self.fc_emb(notes))
        notes = self.norm2(notes)
        notes=notes.unsqueeze(1)
        x=x.unsqueeze(1)
        x,_= self.attention(notes, x, x)
        x = self.fc2(x)
        x = x.squeeze(1)
        return x

class MMCNN_SUM_ATT(nn.Module):
    def __init__(self, ecg_channels=12):
        super(MMCNN_SUM_ATT, self).__init__()
        # ECG processing Layers
        self.name = "MMCNN_SUM_ATT"
        self.conv1 = nn.Conv1d(ecg_channels, 16, 7)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(32, 48, 3)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.fc0 = nn.Linear(5856, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 5)

        # Clinical Notes Processing Layers
        self.fc_emb = nn.Linear(768, 128)
        self.norm = nn.LayerNorm(128)

        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.activation = nn.ReLU()

    def forward(self, x, notes):
        # ECG Processing
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))

        # Notes Processing
        notes = notes.view(notes.size(0),-1)
        notes = self.activation(self.fc_emb(notes))
        x = self.norm(x + notes)

        x=x.unsqueeze(1)
        # print(x.shape)
        x,_= self.attention(x, x, x)
        
        x = self.fc2(x)
        x = x.squeeze(1)
        return x

if __name__ == "__main__":
    model = MMCNN_ATT()
    