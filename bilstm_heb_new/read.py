import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#预测每个站的流量是模型的主要目的，如果要按照
def initialize(sequence_length, batch_size, filepath=None, sheet_name=None, skiprows=0, skipfooter=0):
    file_name = filepath
    assert isinstance(file_name, str), "filepath must be a string"
    data = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=skiprows, skipfooter=skipfooter,
                         index_col=0)
    data.fillna(method='ffill', inplace=True)  # 缺失值补齐
    data_array = data.to_numpy()

    # 创建 MinMaxScaler 对象
    scaler = MinMaxScaler()
    # 拟合数据并进行归一化
    normalized_data_array = np.hstack((data_array[:,0].reshape(-1, 1),scaler.fit_transform(data_array[:,1].reshape(-1, 1))))
    tensor_data = torch.tensor(normalized_data_array, dtype=torch.float32)

    train_data = ExcelDataset(sequence_length, tensor_data[:int(len(tensor_data) * 0.8)])
    test_data = ExcelDataset(sequence_length, tensor_data[int(len(tensor_data) * 0.8):])
    train_iter = DataLoader(train_data, batch_size)
    test_iter = DataLoader(test_data, batch_size)
    return train_iter, test_iter, scaler


class ExcelDataset(Dataset):
    def __init__(self, sequence_length,torch_file):
        self.tensor_data=torch_file
        self.sequence_length=sequence_length
    def __getitem__(self, idx):
        x=self.tensor_data[idx:idx+self.sequence_length]
        y=self.tensor_data[idx+self.sequence_length]
        return x.float(),y.float()
    def __len__(self):
        return len(self.tensor_data)-self.sequence_length
