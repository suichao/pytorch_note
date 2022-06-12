# 欧式相似度
import torch
a = torch.rand([1, 10])
b = torch.rand([1, 10])
sim = lambda x, y: (x - y).pow(2).mean().sqrt()
print(sim(a, b))


# 余弦相似度
c_sim = lambda x, y: sum(torch.mul(x, y))/(x.pow(2).sqrt() * y.pow(2).sqrt())
print(c_sim(a, b))