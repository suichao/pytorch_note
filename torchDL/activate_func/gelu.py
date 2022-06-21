import torch
import matplotlib.pyplot as plt

gelu = torch.nn.GELU()
input = torch.linspace(-6, 6, 100)
output = gelu(input)

# 可视化激活函数
plt.plot(input.data.numpy(), output.data.numpy(), "r-")
plt.title("gelu")
plt.grid()

plt.show()