import torch
import torch.nn as nn
import torch.optim as optim
import read as rd
import matplotlib.pyplot as plt
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 2)  # 输出层改为 2

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM 输出
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc1(out)
        out = torch.relu(out)  # 使用 ReLU 激活函数
        out = self.fc2(out)
        out = torch.relu(out)  # 使用 ReLU 激活函数
        out = self.fc3(out)  # 最后输出
        return out

# 超参数
input_size = 2  # 输入特征的数量
hidden_size = 256  # LSTM 隐藏层的大小
output_size = 2  # 输出特征的数量（修改为 2）
num_layers = 4  # LSTM 层数
num_epochs = 100  # 训练轮数
learning_rate = 0.001  # 学习率
sequence_length = 7
batch_size = 30

# 创建模型、损失函数和优化器
train_iter, test_iter, scaler = rd.initialize(sequence_length, batch_size, filepath='test/new_sheet.xlsx', sheet_name=0)
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for x_train, y_train in train_iter:
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 清空梯度

        # 前向传播
        outputs = model(x_train)
        loss = criterion(outputs, y_train)  # 计算损失

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = running_loss / len(train_iter)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 测试模型
model.eval()  # 设置模型为评估模式
test_loss = 0.0
all_predictions = []
all_labels = []
with torch.no_grad():
    for x_test, y_test in test_iter:
        prediction = model(x_test)
        loss = criterion(prediction, y_test)
        test_loss += loss.item()
        all_predictions.extend(prediction.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

avg_test_loss = test_loss / len(test_iter)
print(f'Test Loss: {avg_test_loss:.4f}')

# 反归一化和绘图部分保持不变
# 反归一化
all_predictions = np.array(all_predictions)  # 转换为 NumPy 数组
all_labels = np.array(all_labels)

# 假设 scaler 是 MinMaxScaler 对象，且数据是多变量的，这里需要调整形状以适应反归一化
all_predictions_reshaped = all_predictions.reshape(-1, 2)  # 修改为 (N, 2)
all_labels_reshaped = all_labels.reshape(-1, 2)  # 修改为 (N, 2)

# 反归一化
inverse_predictions = scaler.inverse_transform(all_predictions_reshaped)
inverse_labels = scaler.inverse_transform(all_labels_reshaped)

# 绘制对比曲线图
plt.figure(figsize=(12, 6))

# 绘制真实数据
plt.plot(inverse_labels[:, 0], label='True Data - Feature 1', color='blue')
plt.plot(inverse_labels[:, 1], label='True Data - Feature 2', color='orange')

# 绘制预测数据
plt.plot(inverse_predictions[:, 0], label='Predicted Data - Feature 1', linestyle='dashed', color='green')
plt.plot(inverse_predictions[:, 1], label='Predicted Data - Feature 2', linestyle='dashed', color='red')

plt.title('True vs Predicted Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()




all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
# 假设 scaler 是 MinMaxScaler 对象，且数据是单变量的，这里需要调整形状以适应反归一化
all_predictions_reshaped = all_predictions.reshape(-1, 1)
#适应原数组情况
predictions_with_dummy_feature = np.hstack((all_predictions_reshaped, np.zeros((all_predictions_reshaped.shape[0], 1))))


all_labels_reshaped = all_labels.reshape(-1, 1)

labels_with_dummy_feature = np.hstack((all_labels_reshaped, np.zeros((all_predictions_reshaped.shape[0], 1))))

inverse_predictions = scaler.inverse_transform(predictions_with_dummy_feature)

final_predictions = inverse_predictions[:, 0]
print(inverse_predictions)
inverse_labels = scaler.inverse_transform(labels_with_dummy_feature)
print(inverse_labels)
final_labels = inverse_labels[:, 0]