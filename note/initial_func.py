import torch
tensor = torch.nn.Parameter()
# 1. 均匀分布
# 服从~U ( a , b ) U(a, b)U(a,b)
torch.nn.init.uniform_(tensor, a=0, b=1)


# 2. 正太分布
# 服从~N ( m e a n , s t d ) N(mean, std)N(mean,std)
torch.nn.init.normal_(tensor, mean=0, std=1)


# 3. 初始化为常数
# 初始化整个矩阵为常数val
val = 1
torch.nn.init.constant_(tensor, val)


# 4. Xavier
# 基本思想是通过网络层时，输入和输出的方差相同
# 如果初始化值很小，那么随着层数的传递，方差就会趋于0，此时输入值 也变得越来越小，在sigmoid上就是在0附近，接近于线性，失去了非线性
# 如果初始值很大，那么随着层数的传递，方差会迅速增加，此时输入值变得很大，而sigmoid在大输入值写倒数趋近于0，反向传播时会遇到梯度消失的问题
# 对于Xavier初始化方式，pytorch提供了uniform和normal两种：
# 均匀分布 ~ U(−a , a)
torch.nn.init.xavier_uniform_(tensor, gain=1)

# 正态分布~N ( 0 , std)
torch.nn.init.xavier_normal_(tensor, gain=1)