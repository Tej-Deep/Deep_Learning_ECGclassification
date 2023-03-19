import pandas as pd
from torch.utils.data import Dataset

class PTBDataset(Dataset):
    def __init__(self, X, df, Y):
        self.labels = Y
        self.data = X
        self.id = df.index.to_list()
        self.notes = df.report.to_list()

    def __getitem__(self, i):
        return self.data[i] , self.labels[i], self.notes[self.id[i]]

    def __len__(self):
        return len(self.labels)