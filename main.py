import torch

x=torch.arange(4.0)
x.requires_grad_(True)#开启x的梯度存储
# 创建一个张量并设置为需要梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 计算一个非标量的输出

y = x * x  # y 是一个张量，形状为 (3,)
u=y.detach()
y=u*x
# 定义一个与 y 形状相同的梯度张量
grad_y = torch.tensor(y)  # 每个元素的梯度

# 反向传播，使用 gradient 参数
y.backward(gradient=grad_y)

# 输出 x 的梯度
print(x.grad)  # 输出: tensor([0.2000, 2.0000, 1.0000])