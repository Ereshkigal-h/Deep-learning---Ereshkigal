import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
trans = transforms.ToTensor()  # 将图像转换为pytorch的 的 Tensor 格式
mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)  # 加载训练集
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)  # 加载测试集


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# X,y=next(iter(data.DataLoader(mnist_train,batch_size=18)))#创建一个生成器，每次返回一个训练组
# print(y,get_fashion_mnist_labels(y))查看数据集数据组成
# show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))生成图像
# plt.show()本地编译器图像显示
batch_size = 256


def get_dataloader_workers():
    return 2


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())  # 也是构建生成器，shuffle是随机顺序，num_workers是指定读取核数这边是指定4核
'''
timer = d2l.Timer()  # 读取时间测试，读取时间要尽量小于测试时间
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
'''

