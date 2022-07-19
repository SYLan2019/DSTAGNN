import torch
import numpy as np
import pandas as pd

def load_weighted_adjacency_matrix(file_path, num_v):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df > 0)
    return df

def load_PA(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.float64(df>0)
    return df

def load_weighted_adjacency_matrix2(file_path, num_v):
    df = pd.read_csv(file_path, header=None)
    df = df.to_numpy()
    df = np.int64(df > 0)
    id_mat = np.identity(num_v)
    mydf = df - id_mat
    return mydf

def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, day_slot, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_pred, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i, :, :] = data[tail: tail + n_pred].reshape(n_pred, n_vertex)

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)