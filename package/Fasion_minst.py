import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def get_dataloader_workers():  # cpu核数确定
    return 2


def load_data_fasion_minst(batch_size, resize=None):
    trans = [transforms.ToTensor()]  # 列表如果有后续操作可以直接写列表里面,这里的操作是成为一个张量
    if resize:  # python中，只要非空就视为真
        trans.insert(0, transforms.Resize(resize))  # 在train列表之前加上一个变形，他会按顺序执行的
    trans = transforms.Compose(trans)  # 组合计算之前trans列表里的组合，使其成为一个可用对象
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))#返回两个迭代器第一个是训练集的，第二个是测试集
#一个迭代器中分别两个张量生成x,y
