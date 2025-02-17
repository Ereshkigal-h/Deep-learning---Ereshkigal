# 解释器配置在root@120.27.149.170.22
# 线性回归的从零开始实现
import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 用于生成一个符合高斯分布的向量 0,1是均值和方差，后面是总样本数和实列样本数（一次计算里面的参数）
    y = torch.matmul(X, w) + b  # marmul用来计算矩阵乘积，这一步其实就是计算线性函数的值
    y + torch.normal(0, 0.01, y.shape)  # 添加浮动
    return X, y.reshape((-1, 1))  # 返回的是每个值都是一个训练实例（向量）的向量和每个训练实例的扰动结果


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 100)  # features是x矩阵（1000行2列）labels是1000行y


def data_iter(batch_size, features, labels):  # 生成小批量数据
    num_examples = len(features)  # 读取features的行数，行数也就是样本数量
    indices = list(range(num_examples))  # 把num_examples变成列表
    random.shuffle(indices)  # 打乱下标
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])  # 将数据集换成一段一段的方式，这里里面存的都是下标
        yield features[batch_indices], labels[batch_indices]  # 建立一个生成器（迭代器的一种），逐步返回值,返回一个向量x和对应的y


bash_size = 10
# for x,y in data_iter(bash_size,features,labels):#这个时候x,y是从训练集里面随机取出来的结果

w = torch.normal(0, 0.00, size=(2, 1), requires_grad=True)  # 这个梯度是两行一列，也就是w1，w2
print(w)
b = torch.zeros(1, requires_grad=True)  # 常数向量


def linreg(X, w, b):  # 计算预测值
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):  # 均方损失,y_hat是预测值
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):  # params是参数列表（包含w和b）之后将每个参数选一遍做梯度下降算法
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 梯度下降算法，lr是学习率，batch/size前面没有除现在除实际上是包含在param.grad里面的
            #param -= lr * param.grad / batch_size 会同时更新 param 中的所有元素根据每个元素不同的梯度进行更新，所以是元素级操作
            param.grad.zero_()  # 梯度清零计算


# 训练过程
lr = 1#学习率可以调，有的时候训练效果指数级提升
num_epochs = 4

for epoch in range(num_epochs):
    for X, y in data_iter(bash_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)  # 小批量损失
        l.sum().backward()#这里是求平均训练损失关于w的梯度(除以n在sgd中有),这里会自动锁定开启grad的参数并保存在.grad中
        sgd([w,b],lr,bash_size)
    with torch.no_grad():
        train_l=squared_loss(linreg(features,w,b),labels)#最后计算所有样本的损失并取平均值，取平均值用的mean()
        print(f'epoch {epoch+1},loss {float(train_l.mean()):f}')
        print(w)