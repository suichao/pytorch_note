import torch
from torch import nn
from torch import Tensor


class ConditionLayerNormalization(nn.Module):
    """
    自适应层归一化
    """
    def __init__(self, condition_hidden_size, hidden_size, eps=1e-12):
        super(ConditionLayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dense1 = nn.Linear(condition_hidden_size, hidden_size, bias=False)
        self.dense1.weight.data.uniform_(0, 0)
        self.dense2 = nn.Linear(condition_hidden_size, hidden_size, bias=False)
        self.dense2.weight.data.uniform_(0, 0)  # 全零初始化
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs: Tensor, condition_inputs):
        assert inputs.shape[-1] == self.hidden_size
        u = inputs.mean(-1, keepdim=True)
        s = (inputs - u).pow(2).mean(-1, keepdim=True)
        o = (inputs - u)/torch.sqrt(s + self.eps)

        for _ in range(len(inputs.shape) - len(condition_inputs.shape)):
            condition_inputs = condition_inputs.unsqueeze(dim=1)

        c_w = self.dense1(condition_inputs)
        c_b = self.dense2(condition_inputs)

        return (self.weight + c_w) * o + (self.bais + c_b)
