import torch
a = torch.tensor([1, 2, 3])
b = a.unsqueeze(1)
print(b)
b = b.squeeze(1)
print(b)