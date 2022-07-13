import numpy as np
# import tensorflow as tf
from loss import KlLoss, CrossEntropyLoss, BinaryCrossEntropyLoss
import torch

x = [np.random.randint(1, 11) for _ in range(10)]
px = x / np.sum(x)
px = torch.tensor([px])
y = [np.random.randint(1, 11) for _ in range(10)]
py = y / np.sum(y)
py = torch.tensor([py])
res = KlLoss()(px, py)
print(res)
c_res = CrossEntropyLoss()(px, py)
print(c_res)
b_res = BinaryCrossEntropyLoss()(px, py)
print(b_res)
ty_res = torch.nn.CrossEntropyLoss()(px, py)
print(ty_res)