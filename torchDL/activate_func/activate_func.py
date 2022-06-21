import torch
import torch.nn as nn
import matplotlib.pyplot as plt


x = torch.linspace(-6, 6, 100)
# sigmoid激活函数
sigmoid = nn.Sigmoid()
ysigmoid = sigmoid(x)

# Tanh激活函数
tanh = nn.Tanh()
ytanh = tanh(x)

# ReLU激活函数
relu = nn.ReLU()
yrelu = relu(x)

# Softplus激活函数
softplus=nn.Softplus()
ysoftplus = softplus(x)

# 可视化激活函数
plt.figure(figsize=(14, 3))
plt.subplot(1, 4, 1)
plt.plot(x.data.numpy(), ysigmoid.data.numpy(), "r-")
plt.title("Sigmoid")
plt.grid()

plt.subplot(1, 4, 2)
plt.plot(x.data.numpy(), ytanh.data.numpy(), "r-")
plt.title("Tanh")
plt.grid()

plt.subplot(1, 4, 3)
plt.plot(x.data.numpy(), yrelu.data.numpy(), "r-")
plt.title("ReLU")
plt.grid()

plt.subplot(1, 4, 4)
plt.plot(x.data.numpy(), ysoftplus.data.numpy(), "r-")
plt.title("Softplus")
plt.grid()
plt.show()