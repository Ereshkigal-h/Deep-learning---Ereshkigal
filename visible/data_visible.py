import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_excel("2020哈尔滨轨道交通站点客流数据-0706.xlsx", sheet_name=7)
print(data)
data = data.to_numpy()
data = data
# 创建 MinMaxScaler 对象
scaler = MinMaxScaler()
# 拟合数据并进行归一化
normalized_data_array = scaler.fit_transform(data)
data = normalized_data_array.T

# 标签
labels = [r"people",r'data']
print(data.shape[0])
# 绘制每一行
for i in range(data.shape[0]):
    plt.plot(data[i], label=labels[i])  # 使用 label 参数

plt.title('Line Plot of Each Row')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()  # 在循环外调用
plt.show()