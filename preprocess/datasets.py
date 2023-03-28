import pandas as pd
from torch.utils.data import Dataset
import torch

class PTBDataset(Dataset):
    def __init__(self, X, df, Y, embed_path='./data/embeddings/'):
        self.labels = Y
        self.data = X
        # self.id = df.index.to_list()
        self.notes = df.report.to_list()
        self.embed_path = embed_path

    def __getitem__(self, i):
        return self.data[i] , self.labels[i], torch.load(f'{self.embed_path}{self.id[i]}.pt')

    def __len__(self):
        return len(self.labels)