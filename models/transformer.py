import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


class Embedder(nn.Module):
    '''Input embedding layer of size vocab_size * dimensionality
    of word embedding'''
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self,x):
        return self.embed(x.long())

class pre_conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conven = nn.Conv2d(12,1,1)
    
    def forward(self, src):
        src[src < 0] = 0
        src[src > 249] = 249
        src = src.float().unsqueeze(3)
        print(src.shape)
        src = self.conven(src)
        src = torch.squeeze(src,2)
        return src

class PositionalEncoding(nn.Module):
    '''Transformers are not sequential so positional encoding
    gives some sequentiality to sentence'''

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x *= math.sqrt(self.d_model)
        x +=  self.pe[:,:x.size(1)].cuda()
        return self.dropout(x)

class Classifier_CNN(nn.Module):
    '''Convolutional output network for classification task - takes in summed input sequence and decoder output'''
    def __init__(self):
        super(Classifier_CNN, self).__init__()
        self.name = "cnn"
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 1, 3)
        self.conv3 = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(377, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16,5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

class MVMNet_Transformer(nn.Module):
    def __init__(self,  vocab_size, d_model, num_layers, heads) -> None:
        super().__init__()
        self.pre_CNN= pre_conv()
        self.pe = PositionalEncoding(d_model)
        self.embed = Embedder(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model = d_model,nhead = heads,num_decoder_layers=num_layers, num_encoder_layers=num_layers)
        self.clasifier = Classifier_CNN()
        self.transform_cnn = torchvision.transforms.Resize((250,120))
        self.norm = nn.LayerNorm((1,250,120))
    def forward(self, x, tgt=torch.rand((250,120))):
        x = ((x + 1.68)*100).long()
        out = self.pe(self.embed(self.pre_CNN(x)))

        out = self.transformer(src=out, tgt=tgt)
        out = torch.squeeze(out,0)

        x = self.transform_cnn(x.float())
        
        out = self.clasifier(self.norm(out+x))
        return out

