import matplotlib.pyplot as plt
import torch
from IPython import display
from d2l import torch as d2l
from package.Fasion_minst import *


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def soft_max(x):  # 将处理出的y-hat转换成概率形式
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition  # 用了广播机制


def cross_entorpy(y_hat, y):  # loss
    # y-hat的每一行是10数，结果是10个不同的输出组成的一组结果，每一行是一个样本
    return -torch.log(y_hat[range(len(y_hat)), y])  # y 张量包含每个样本的真实类别索引，这里只是提取出y_hat的真实值，y的真实值恒为1


# len(y_hat) 的含义是获取这个矩阵的第一维的大小，也就是行数
def accuracy(y_hat, y):  # 用来表示精度使用的函数，查看y_hat概率最高的一项和y下标是否一致
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 保证是个二维张量，说实话有点迷惑
        y_hat = y_hat.argmax(axis=1)  # 按列求最大值下标存到y_hat里
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())  # 这边sum可以计算true的数量


# 找到预测正确的样本数，之后结合长度，就可以求出预测准确的概率
def evaluate_accuracy(net, data_iter):  # 评估指定模型的精度
    if isinstance(net, torch.nn.Module):  # isinstance() 是一个内置函数，用于检查一个对象是否是某个类或其子类的实例
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)
    for x, y in data_iter:
        metric.add(accuracy(soft_max(net(x)), y), y.numel())  # 前面是预测正确的y-hat，后面是y真实值的长度
    return metric[0] / metric[1]  # 除一下就可以评估预测模型的预测成功概率


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):  # 使用pytorch函数的情况
        net.train()
        # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  # 使用pytorch函数的情况
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 训练损失总和、训练准确度总和、样本数
    return metric[0] / metric[2], metric[1] / metric[2]  # 第一个是训练损失除以总样本量，因为是log（pi)pi接近1的时候总的接近0，所以越接近0越好，第二个是模型精确率


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save#是上面全部的主函数，同时有训练和评估功能
    """训练模型（定义见第3章）"""
    #animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        #legend=['train loss', 'train acc', 'test acc'])  # 动画实现，不用知道
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # 进行训练
        test_acc = evaluate_accuracy(net, test_iter)  # 进行评估
        print(train_metrics,test_acc)
        # animator.add(epoch + 1, train_metrics + (test_acc,))# 动画演示
    train_loss, train_acc = train_metrics  # 得出结果
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc
