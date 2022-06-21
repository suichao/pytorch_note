import torch
import torch.nn as nn
import matplotlib.pyplot as plt
"""
  使用这个函数作为激活函数有如下几个特点：
   1）tanh 类似于幅度增大 sigmoid，将输入值转换为 -1 至 1 之间。tanh 的导数取值范围在 0 至 1 之间，优于 sigmoid 的 0 至 1/4，在一定程度上，减轻了梯度消失的问题。
   2）tanh 是 0 均值的，在实际应用中 tanh 会比 sigmoid 更好。
"""
x = torch.linspace(-6, 6, 100)

# Tanh激活函数
tanh = nn.Tanh()
ytanh = tanh(x)
plt.subplot(1, 4, 2)
plt.plot(x.data.numpy(), ytanh.data.numpy(), "r-")
plt.title("Tanh")
plt.grid()
plt.show()