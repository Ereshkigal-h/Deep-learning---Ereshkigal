import torch
from torch import nn
from d2l import torch as d2l
from package.Fasion_minst import *
from package.ch3 import train_ch3

batch_size = 256
train_iter, test_iter = load_data_fasion_minst(batch_size)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))  # torch.Size([256, 1, 28, 28])数据格式


# Flatten()只保留第一维度，后面全部展平
def init_weights(m):  # 正态分布初始化权重
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)  # PyTorch 中用于初始化神经网络模型中所有层的参数的语句
loss = nn.CrossEntropyLoss()  # 直接就是交叉熵损失函数，轮椅啊
trainer = torch.optim.SGD(net.parameters(),lr=0)  # parameters(): 这是 nn.Module 类中的一个方法，用于返回模型中所有可学习的参数（如权重和偏置）的迭代器。它会遍历模型的每一层，并收集所有的参数。
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 引用之前的函数
