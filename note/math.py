import torch

# 求平均
a = torch.tensor([3, 4, 5], dtype=torch.float64)
# 保持维度
b = a.mean(-1, keepdim=True)
c = torch.mean(a)
print('均值', b)
print('均值', c)

# 求和
print('求和', a.sum())
print('求和', torch.sum(a))

# 平方
print('平方值:', a.pow(2))
print('平方值:', torch.pow(a, 2))

# 开方
print('开方', torch.sqrt(a))
print('开方', a.sqrt())

# 张量对应元素相乘
b = a
c = torch.mul(a, b)
print("元素相乘：", c)

d = torch.matmul(a, b.t())
print("内积:", d)

# 加法
print("加：", a.add(b))
print("加：", a + b)

# 减法
print("减:", a.sub(a))
print("减:", a - a)

# 除
print("除：", a.div(b))
print("除：", a / b)