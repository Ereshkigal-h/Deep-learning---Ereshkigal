import torch
from torch import nn
from d2l import torch as d2l
#这个计算损失
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05#计算公式就是这个
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
#这一段是随机生成一个数据集来测试
def init_params():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return w,b
def l2_penalty(w):
    return torch.sum(w.pow(2))/2
def train(lambd):
    w,b=init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss#一行自定义
    num_epochs,lr=100,0.03
    for epochs in range(num_epochs):
        for X,y in train_iter:
            l=loss(net(X),y)+lambd*l2_penalty(w)
            l.mean().backward()
            d2l.sgd([w,b],lr,batch_size)
        print(l.mean(),w-true_w)
train(2)
