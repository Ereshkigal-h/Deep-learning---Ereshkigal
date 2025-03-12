import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 清空环境（Python 中不需要特意清空变量，通常使用脚本）
# 关闭警告
import warnings
warnings.filterwarnings('ignore')

# 导入数据
res = pd.read_excel('客流.xlsx').values

# 数据分析
num_size = 0.8  # 训练集占数据集比例
outdim = 1  # 最后一列为输出
num_samples = res.shape[0]  # 样本个数
# 打乱数据集
# np.random.shuffle(res)  # 如果需要打乱数据，取消注释这一行
num_train_s = int(num_size * num_samples)  # 训练集样本个数
f_ = res.shape[1] - outdim  # 输入特征维度

# 划分训练集和测试集
P_train = res[:num_train_s, :f_]
T_train = res[:num_train_s, f_:]
P_test = res[num_train_s:, :f_]
T_test = res[num_train_s:, f_:]

# 数据归一化
scaler_input = MinMaxScaler(feature_range=(0, 1))
p_train = scaler_input.fit_transform(P_train)
p_test = scaler_input.transform(P_test)

scaler_output = MinMaxScaler(feature_range=(0, 1))
t_train = scaler_output.fit_transform(T_train)
t_test = scaler_output.transform(T_test)

# 转换为 PyTorch 张量
vp_train = [torch.tensor(p_train[i], dtype=torch.float32).view(-1, 1) for i in range(p_train.shape[0])]
vt_train = [torch.tensor(t_train[i], dtype=torch.float32).view(-1, 1) for i in range(t_train.shape[0])]

vp_test = [torch.tensor(p_test[i], dtype=torch.float32).view(-1, 1) for i in range(p_test.shape[0])]
vt_test = [torch.tensor(t_test[i], dtype=torch.float32).view(-1, 1) for i in range(t_test.shape[0])]

# 输出数据格式（可选）
print("训练集输入数据：", vp_train)
print("训练集输出数据：", vt_train)
print("测试集输入数据：", vp_test)
print("测试集输出数据：", vt_test)
self.data = data = pd.read_excel(filepath, sheet_name=0, skiprows=1, index_col=0)