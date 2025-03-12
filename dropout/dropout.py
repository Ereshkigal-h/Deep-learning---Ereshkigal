import torch
from torch import nn
from d2l import torch as d2l
from package.Fasion_minst import *
from package.ch3 import  *
def dropout_layer(X,dropout):#dropout也是一个超参数
    assert  0 <= dropout <=1
    if dropout==1:#全丢了，丢弃概率100%
        return torch.zeros_like(X)
    if dropout==0:#一个不丢
        return X
    mask=(torch.randn(X.shape)>dropout).float()#调用一个随机函数生成一个随机张量，最后和dropout比较，转变成0,1浮点数
    return mask*X /(1.0-dropout)#做相乘，如果mask是0舍去，1就做比例放大维持期望
#乘法gpu算的比选取快
'''
#测试函数
X=torch.arange(16,dtype=torch.float32).reshape((2,8))
print(X)
print(dropout_layer(X,0.))
print(dropout_layer(X,1))
print(dropout_layer(X,0.1))
'''
#定义训练网络层
num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256
dropout1,dropout2=0.2,0.5
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.training=is_training
        self.lin1=nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu=nn.ReLU()
    def forward(self,X):
        H1=self.relu(self.lin1(X.reshape(-1,self.num_inputs)))#行是样本数量，列是输入个数
        if self.training == True:
            H1=dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2=dropout_layer(H2,dropout2)
        out= self.relu(self.lin3(H2))
        return out
net=Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)
num_epochs,lr,batch_size=10,0.5,256
train_iter,test_iter=load_data_fasion_minst(256)
loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=lr)
train_ch3(net,train_iter, test_iter, loss, num_epochs, trainer)
