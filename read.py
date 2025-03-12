import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas
#预测每个站的流量是模型的主要目的，如果要按照

class ExcelDataset(Dataset):
    def __init__(self, filepath=None, sheet_name=None, skiprows=None, skipfooter=0):
        self.file_name = filepath
        assert (self.file_name is not None) or type(self.file_name) == str
        self.data = pandas.read_excel(filepath, sheet_name=sheet_name, skiprows=skiprows, skipfooter=skipfooter,
                                      index_col=0)
        self.data.fillna(method='ffill', inplace=True)#缺失值补齐
        data_array = self.data.to_numpy()
        tensor_data = torch.tensor(data_array)
        self.tensor_data = tensor_data#竖列是每一个车站一年的变化

        #print(tensor_data)
    def get_data(self):
        return self.tensor_data
    def __iter__(self):
        i=0
        while i <len(self.tensor_data):
            yield self.tensor_data[i:i+7]
            i+=7
    def __len__(self):
        return len(self.tensor_data)
a = ExcelDataset('new_sheet.xlsx', 0, skiprows=0, skipfooter=0)

