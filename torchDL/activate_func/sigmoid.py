"""
Sigmoid 函数是应用最广泛的非线性激活函数之一，它可以将值转换为 $0$ 和 $1$ 之间，如果原来的输出具有这样的特点：值越大，归为某类的可能性越大，
那么经过 Sigmod 函数处理的输出就可以代表属于某一类别的概率。其数学表达式为：
import torch

S = torch.nn.Sigmoid()  # 这是一个类。需要初始化后才能使用。
x = torch.randn(2)
y = S(x)  # 元素层面的操作
print(y)

从数学上来看，非线性的 $Sigmoid$ 函数对中央区的信号增益较大，对两侧区的信号增益小，在信号的特征空间映射上，有很好的效果。
从神经科学上来看，中央区酷似神经元的兴奋态，两侧区酷似神经元的抑制态，因而在神经网络学习方面，可以将重点特征推向中央区，
将非重点特征推向两侧区。无论是哪种解释，看起来都比早期的线性激活函数, 阶跃激活函数高明了不少。

sigmoid在深度学习应用较少，主要是由于函数求导之后，从图像来看左右两端趋向零，不利于梯度更新，容易出现梯度消失
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


x = torch.linspace(-6, 6, 100)
# sigmoid激活函数
sigmoid = nn.Sigmoid()
ysigmoid = sigmoid(x)


plt.plot(x.data.numpy(), ysigmoid.data.numpy(), "r-")
plt.title("Sigmoid")
plt.grid()
plt.show()