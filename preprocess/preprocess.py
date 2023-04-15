import pandas as pd
import numpy as np
import wfdb
import ast

from collections import Counter
from torch.utils.data import DataLoader

from sklearn.preprocessing import MultiLabelBinarizer

from .datasets import PTBDataset


def load_raw_data_ptbxl(df: pd.core.frame.DataFrame, sampling_rate: int, path: str):
    if sampling_rate == 100:
        # Loading all data with signal and meta information
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def load_dataset(path: str, sampling_rate: int = 100):
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    return X, Y


def compute_label_agg(df: pd.core.frame.DataFrame, path: str):
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp))

    df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
    df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))

    return df


def select_data(XX, YY, min_samples: int = 0):
    """
    Selects data and converts labels to multi-hot
    """
    mlb = MultiLabelBinarizer()
    counts = pd.Series(np.concatenate(
        YY.superdiagnostic.values)).value_counts()
    counts = counts[counts > min_samples]
    YY.superdiagnostic = YY.superdiagnostic.apply(
        lambda x: list(set(x).intersection(set(counts.index.values))))
    YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
    X = XX[YY.superdiagnostic_len > 0]
    Y = YY[YY.superdiagnostic_len > 0]
    y = mlb.fit_transform(Y.superdiagnostic.values)

    return X, Y, y


def sample_class(data, labels, Y, cls='HYP'):
    mask = labels['superdiagnostic'].apply(lambda x: cls in x)

    return data[mask], labels[mask], Y[mask]


def no_to_augment(df, cls):
    """Min and max difference between cls and other classes. 

    Args:
        df (Dataframe): Dataframe of labels
        cls (str): Class to augment
    """

    counts = df['superdiagnostic'].apply(lambda x: Counter(x))
    c = sum(counts, Counter())
    differences = [abs(c[cls] - c[k]) for k in c.keys() if k != cls]

    # find the maximum and minimum differences
    max_diff = max(differences)
    min_diff = min(differences)

    return max_diff, min_diff


def undersample(data, df, Y, cls='NORM'):
    """Attempts to undersample the class cls to match the number of samples in the second most common class.
    Only removes samples that belong solely to the class cls.
    Will not work if samples of only class cls is less than the difference between the number of samples of the most common class and the second most common class.
    If we remove samples that belong to multiple classes, we will lose information about the other classes.

    Args:
        cls (str, optional): Class to undersample. Defaults to 'NORM'.
    """
    df = df.copy()
    counts = df['superdiagnostic'].apply(lambda x: Counter(x))
    total_counts = sum(counts, Counter())
    keep = total_counts.most_common(2)[-1][1]

    df.loc[df.loc[:, 'superdiagnostic'].apply(
        lambda x: cls in x), 'is_cls'] = True
    df.loc[df.loc[:, 'superdiagnostic'].apply(
        lambda x: cls == x[0] and len(x) == 1), 'is_only_cls'] = True
    norm_counts = df['is_cls'].value_counts()
    undersample_count = norm_counts - keep

    mask = ~((df['is_only_cls']) & (df.groupby(
        'is_only_cls').cumcount() < undersample_count[True]))

    return data[mask], df[mask], Y[mask]


def get_data_loaders(data, labels, Y, batch_size: int, ratio: list[float] = [0.8, 0.1, 0.1], shuffle_data = True):
    """
    Get train, validation and test DataLoaders

    Returns: 
        (DataLoader, DataLoader, DataLoader): Train, Validation, Test DataLoaders, in that order. 
    """
    # 1-8 for training
    X_train = data[labels.strat_fold < 9]
    y_train = Y[labels.strat_fold < 9]
    df_train = labels[labels.strat_fold < 9]
    train = PTBDataset(X_train, df_train, y_train)
    # 9 for validation
    X_val = data[labels.strat_fold == 9]
    y_val = Y[labels.strat_fold == 9]
    df_val = labels[labels.strat_fold == 9]
    val = PTBDataset(X_val, df_val, y_val)
    # 10 for test
    X_test = data[labels.strat_fold == 10]
    y_test = Y[labels.strat_fold == 10]
    df_test = labels[labels.strat_fold == 10]
    test = PTBDataset(X_test, df_test, y_test)

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle = shuffle_data)
    valid_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle = shuffle_data)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle = shuffle_data)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Example
    path = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    batch_size = 265
    print("Start loading")
    data, raw_labels = load_dataset(path)
    print("done loading")
    labels = compute_label_agg(raw_labels, path)

    data, labels, Y = select_data(data, labels)

    train_loader, valid_loader, test_loader = get_data_loaders(
        data, labels, Y, batch_size)

    # for batch_number, batch in enumerate(train_loader):
    #     inputs, outputs, text = batch
    #     print("---")
    #     print("Batch number: ", batch_number)
    #     print(inputs)
    #     print(outputs)
    #     # print(text)
    #     print(inputs.shape)
    #     print(outputs.shape)
    #     # print(text.shape)
    #     break
