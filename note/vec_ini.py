import torch

# range
a = torch.range(0, 10, 2)
print(a)
"""
tensor([ 0.,  2.,  4.,  6.,  8., 10.])
"""

# one
b = torch.ones([2, 3])
print(b)


c = torch.Tensor(3, 4)
print('c:', c)