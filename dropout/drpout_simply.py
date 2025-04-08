import torch
from torch import nn
from d2l import torch as d2l
from package.Fasion_minst import *
from package.ch3 import *
dropout1=0.2
dropout2=0.3
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU()
                  ,nn.Dropout(dropout1),nn.Linear(256,256),nn.ReLU(),nn.Dropout(dropout2),
                  nn.Linear(256,256),
                  nn.Linear(256,10))

def __init_weight__(m):
    if type(m)== nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(__init_weight__)
trainer=torch.optim.SGD(net.parameters(),lr=0.5)
train_iter,test_iter=load_data_fasion_minst(256)
loss=nn.CrossEntropyLoss()
train_ch3(net,train_iter,test_iter,loss,num_epochs=10,updater=trainer)