import torch
from torch import nn


class ResidualDgCnn(nn.Module):

    def __init__(self, hidden_dim, kernel_size, dilation_rate):
        super(ResidualDgCnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=kernel_size, dilation=dilation_rate,
                               padding=dilation_rate)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm_layer = nn.LayerNorm()
        self.A = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        c_inputs = torch.permute(inputs, (0, 2, 1))
        c_inputs = self.conv1(c_inputs)
        c_inputs = torch.permute(c_inputs, (0, 2, 1))
        # 拆分为两半，一半用来计算gate
        gate = torch.sigmoid(c_inputs[:, :, :self.hidden_dim])
        c_inputs = c_inputs[:, :, self.hidden_dim:] * gate
        c_inputs = self.norm_layer(c_inputs)
        l_inputs = self.linear(inputs)
        output = l_inputs * (1 - self.A) + c_inputs * self.A
        return output


class DgCnn(nn.Module):

    def __init__(self, hidden_size):
        super(DgCnn, self).__init__()
        self.hidden_size = hidden_size
        self.res_block1 = ResidualDgCnn(self.hidden_size, 3, dilation_rate=1)
        self.res_block2 = ResidualDgCnn(self.hidden_size, 3, dilation_rate=2)
        self.res_block3 = ResidualDgCnn(self.hidden_size, 3, dilation_rate=4)
        self.res_block4 = ResidualDgCnn(self.hidden_size, 3, dilation_rate=1)
        self.res_block5 = ResidualDgCnn(self.hidden_size, 3, dilation_rate=1)

    def forward(self, inputs):
        inputs = self.res_block1(inputs)
        inputs = nn.Dropout(0.1)(inputs)
        inputs = self.res_block2(inputs)
        inputs = nn.Dropout(0.1)(inputs)
        inputs = self.res_block3(inputs)
        inputs = nn.Dropout(0.1)(inputs)
        inputs = self.res_block4(inputs)
        inputs = nn.Dropout(0.1)(inputs)
        inputs = self.res_block5(inputs)
        outputs = nn.Dropout(0.1)(inputs)
        return outputs
