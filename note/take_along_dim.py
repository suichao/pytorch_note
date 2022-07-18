# 接口torch.max将返回值和索引，如何使用索引从另一个张量中获取相应的元素
import torch
a=torch.rand(2,3,4)
b=torch.rand(2,3,4)
indices=torch.max(a,1,keepdim=True)[1]
b_max = torch.take_along_dim(b,indices,dim=1)

a=torch.rand(2,3)
print(a)
# tensor([[0.0163, 0.0711, 0.5564],
#         [0.4507, 0.8675, 0.5974]])
b=torch.rand(2,3)
print(b)
# tensor([[0.7542, 0.1793, 0.5399],
#         [0.2292, 0.5329, 0.2084]])
indices=torch.max(a,1,keepdim=True)[1]
res = torch.take_along_dim(b,indices,dim=1)
# tensor([[0.5399],[0.5329]])
print(res)