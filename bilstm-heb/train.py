import torch
import torch.nn as nn
import torch.optim as optim
import read as rd
import torch
from torch import nn


# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN 输出
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 超参数
input_size = 1  # 输入特征的数量
hidden_size = 32  # RNN 隐藏层的大小
output_size = 1  # 输出特征的数量
num_layers = 7  # RNN 层数
num_epochs = 100  # 训练轮数
learning_rate = 0.001  # 学习率
sequence_length = 20
batch_size = 30
# 创建模型、损失函数和优化器
train_iter, test_iter = rd.initialize(7, batch_size, filepath='new_sheet.xlsx', sheet_name=0)
model = RNNModel(input_size, hidden_size, output_size, num_layers)


def init_weights(m):
    if type(m) == RNNModel:
        for param in m.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)


init_weights(model)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for x_train, y_train in train_iter:
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 清空梯度
        # 前向传播
        outputs = model(torch.softmax(x_train,dim=1))
        loss = criterion(outputs, torch.softmax(y_train,dim=1))  # 计算损失
        # 反向传播和优化
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {double(loss.item())}')

# 测试模型（示例）
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for x_test, y_test in test_iter:
        prediction = model(x_test)
        loss = torch.mean((y_test - prediction) ** 2)
        print("Test prediction:", loss)
