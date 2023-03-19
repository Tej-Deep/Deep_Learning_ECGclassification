import pandas as pd
import numpy as np
import wfdb
import ast

from torch.utils.data import Dataset, DataLoader, random_split

from datasets import PTBDataset

def load_raw_data(df, sampling_rate:int, path:str):
    """
    The first line describes concisely what the functions does. 
    
    Args: 
        argument1 (str): Description of argument1. If you have lots to say about
            a function you can break to a new line and indent the text so it fits. 
        argument2 (int, optional): Description of argument2. 
    
    Returns: 
        str: Optional description explaining the value returned by the function. 
        
    Raises:
        ValueError: Description of any errors or exceptions intentionally raised. 
    
    Notes: 
        Credits: https://physionet.org/content/ptb-xl/1.0.1/example_physionet.py
    """
    if sampling_rate == 100:
        # Loading all data with signal and meta information
        data = [wfdb.rdsamp(path+f)[0] for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array(data)
    return data


def load_with_annotation(path:str, sampling_rate:int=100):
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic, agg_df):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    return Y
    
def get_data_loaders(dataset: Dataset, batch_size:int, ratio: list[float]=[0.8,0.1,0.1]):
    """
    Get train, validation and test DataLoaders 
    
    Args: 
        dataset (Dataset): Dataset to split
        batch_size (int): batch size
        ratio (list[int]): Ratio to split into 
    
    Returns: 
        (DataLoader, DataLoader, DataLoader): Train, Validation, Test DataLoaders, in that order. 
    """
    train, validation, test = random_split(dataset, ratio)
    train_loader = DataLoader(dataset=train, batch_size=batch_size)
    valid_loader = DataLoader(dataset=validation, batch_size=batch_size)                            
    test_loader = DataLoader(dataset=test, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader

if __name__=="__main__":
    path = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    
    X, Y = load_with_annotation(path)
    
    ds = PTBDataset(X, Y)
    
    train_loader, valid_loader, test_loader = get_data_loaders(ds)
    