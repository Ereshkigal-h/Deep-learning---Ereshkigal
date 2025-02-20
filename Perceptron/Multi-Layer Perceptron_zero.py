import torch
from torch import nn
from package.Fasion_minst import *
from package.ch3 import train_ch3
#不知道为什么要起这个名字，感觉就是有隐藏层的网络
batch_size = 256
train_iter, test_iter = load_data_fasion_minst(batch_size)
num_inputs, num_outputs, num_hidden = 784, 10, 150  # 前两个是数据输出，是由28*28的图像和10个类别决定的，后面一个是我们自定义的隐藏层大小
w1 = nn.Parameter(
    torch.randn(num_inputs, num_hidden, requires_grad=True))  # torch.randn 是 PyTorch 中用于生成随机数的一个函数。它用于创建一个指定形状的张量
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))  # nn.Parameter 是 PyTorch 深度学习框架中的一个类，用于定义可学习的参数
w2 = nn.Parameter(torch.randn(num_hidden, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [w1, b1, w2, b2]  # 参数集


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)  # 会比较每个值拟合的


def net(X):
    X = X.reshape((-1,num_inputs))
    y_hat_1=relu(X@w1+b1)
    return y_hat_1@w2+b2#y_hat_2
loss=nn.CrossEntropyLoss()
num_epochs,lr=100,0.01
updater=torch.optim.SGD(params,lr)
train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)