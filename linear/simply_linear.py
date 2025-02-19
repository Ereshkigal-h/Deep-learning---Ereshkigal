# 简洁实现，轮椅启动！
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn  # nn是神经网络缩写


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 有函数直接生成数据集了

batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 创建一个生成器，每次返回一个小训练集
net = nn.Sequential(nn.Linear(2, 1))  # 创建线性单层神经网络
net[0].weight.data.normal_(0, 0.01)  # 逐级访问weight（w）的数据，使用normal_修改数据，normal是规定一个正态分布
net[0].bias.data.fill_(0)  # 找到第一层网络的b来更新b值
loss = nn.MSELoss()  # 自定义均方误差，根据前面定义的自动计算
trainer = torch.optim.SGD(net.parameters(), lr=0.04)  # sgd算法，net.parameters是包含了net内的参数（w，b），这里只是创建优化器并没有实际执行优化
num_epochs = 10
for epochs in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # net(X)是计算X的前向传播输出，这个时候已经输出的是预测值y_hat了
        trainer.zero_grad()  # optimizer.zero_grad() 的作用是清零模型中所有可学习参数的梯度
        l.backward()  # 向后传播
        trainer.step()  # 执行优化
    l = loss(net(features), labels)
    print(f"epoch {epochs + 1} loss {l}")
