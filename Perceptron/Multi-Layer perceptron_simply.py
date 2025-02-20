import torch
from torch import nn
from package.Fasion_minst import *
from package.ch3 import train_ch3
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))  # 我感觉是按顺序执行的

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);  # apply 方法会递归地遍历 net 中的所有子模块，并将每个模块作为参数传递给 init_weights 函数。这样，所有线性层的权重都会被初始化。
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')#如果没有none自动返回平均值，所以之前准确率低
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = load_data_fasion_minst(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
