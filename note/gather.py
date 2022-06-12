import torch
import numpy as np
inputs = torch.tensor(np.array(list(range(9))).reshape(3, 3))
print(inputs)
"""
[[0 1 2]
 [3 4 5]
 [6 7 8]]
"""
index = torch.tensor([[1, 2], [0, 2]])

# dim=0, index为行索引，即具体索引为(1, 0), (2, 1) 和 (0, 0), (2, 1)
a = torch.gather(inputs, dim=0, index=index)
print(a)
"""
[[3, 7],
[0, 7]]
"""

# dim=1, index为列索引，即具体索引为(0, 1), (0, 2) 和 (1, 0), (1, 2)
b = torch.gather(inputs, dim=1, index=index)
print(b)
"""
[[1, 2],
[3, 5]]
"""