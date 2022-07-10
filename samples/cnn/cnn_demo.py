import torch
from torch import nn

conv1 = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=2)
inputs = torch.randn(32, 35, 256)
# batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
inputs = inputs.permute(0, 2, 1)
out = conv1(inputs)

print(out.size())
# torch.Size([32, 100, 34])