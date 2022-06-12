import torch
a = torch.tensor([[1, 2, 3]])
b = a.expand(3, 3)
print(b)
'''
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
'''