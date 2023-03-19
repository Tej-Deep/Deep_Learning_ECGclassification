import pandas as pd
from torch.utils.data import Dataset

class PTBDataset(Dataset):
    def __init__(self, X, Y):
        """
        Docstring goes here
        """
        self.labels = Y
        self.data = X
        # self.id = df.index.to_list()
        # self.notes = df.report.to_list()
        # self.map = {'[\'NORM\']':0,'[\'MI\']':1,'[\'STTC\']':2,'[\'CD\']':3,'[\'HYP\']':4, '[\'HYP\', \'MI\']':5, '[\'HYP\', \'CD\']':6,
        #             '[\'HYP\', \'STTC\']':7,'[\'MI\', \'CD\']':8,'[\'MI\', \'STTC\']':9,'[\'CD\', \'STTC\']':10,'[\'HYP\', \'MI\', \'STTC\']':11,
        #             '[\'HYP\', \'MI\', \'CD\']':12, '[\'MI\', \'CD\', \'STTC\']':13,'[\'HYP\', \'CD\', \'STTC\']':14,'[\'HYP\', \'MI\', \'CD\', \'STTC\']':15, 
        #             '[\'CD\', \'NORM\']':16, '[\'CD\', \'NORM\', \'STTC\']':17, '[\'NORM\', \'STTC\']':18, '[\'HYP\', \'NORM\']':19, '[\'MI\', \'NORM\']':20, 
        #             '[\'HYP\', \'MI\', \'NORM\']':21, '[\'HYP\', \'CD\', \'NORM\']':22, '[\'HYP\', \'MI\', \'CD\', \'NORM\']':23, '[\'HYP\', \'MI\', \'CD\', \'NORM\', \'STTC\']':24,
        #             '[\'HYP\', \'NORM\', \'STTC\']':25, '[\'MI\', \'NORM\', \'STTC\']':26, '[\'HYP\', \'MI\', \'NORM\', \'STTC\']':27, '[\'HYP\', \'CD\', \'NORM\', \'STTC\']':28, '[\'MI\', \'CD\', \'NORM\', \'STTC\']':29}

    def __getitem__(self, i):
        """
        Docstring goes here
        """
        return self.data[i][:250] , self.labels[i]

    def __len__(self):
        """
        Docstring goes here
        """
        return len(self.labels)