import torch
import torch.nn as nn
import math

# https://zhuanlan.zhihu.com/p/442092801


class CosineSimilarity(nn.Module):
    """
    余弦相似度
    """
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


class DotProductSimilarity(nn.Module):

    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result


class ProjectedDotProductSimilarity(nn.Module):
    """
    ProjectedDotProductSimilarity 这个相似度函数做一个投影，然后计算点积，计算公式为：
    xT * w1 * (yT * w2)T
    计算后的激活函数。默认为不激活。
    """

    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class BiLinearSimilarity(nn.Module):
    """
    此相似度函数执行两个输入向量的双线性变换。这个函数有一个权重矩阵“W”和一个偏差“b”，以及两个向量之间的相似度，计算公式为：
    xT*w*y + b
    计算后的激活函数。 默认为不激活。
    """

    def __init__(self, tensor_1_dim, tensor_2_dim, activation=None):
        super(BiLinearSimilarity, self).__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        intermediate = torch.matmul(tensor_1, self.weight_matrix)
        result = (intermediate * tensor_2).sum(dim=-1) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class TriLinearSimilarity(nn.Module):
    """
    此相似度函数执行两个输入向量的三线性变换，计算公式为：
    wT * [x,y,x*y] + b
    计算后的激活函数。 默认为不激活。
    """
    def __init__(self, input_dim, activation=None):
        super(TriLinearSimilarity, self).__init__()
        self.weight_vector = nn.Parameter(torch.Tensor(3 * input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        combined_tensors = torch.cat([tensor_1, tensor_2, tensor_1 * tensor_2], dim=-1)
        result = torch.matmul(combined_tensors, self.weight_vector) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class MultiHeadedSimilarity(nn.Module):
    """
    MultiHeadedSimilarity这个相似度函数使用多个“头”来计算相似度。
    也就是说，我们将输入张量投影到多个新张量中，并分别计算每个投影张量的相似度。这里的结果比典型的相似度函数多一个维度。
    """

    def __init__(self,
                 num_heads,
                 tensor_1_dim,
                 tensor_1_projected_dim=None,
                 tensor_2_dim=None,
                 tensor_2_projected_dim=None,
                 internal_similarity=DotProductSimilarity()):
        super(MultiHeadedSimilarity, self).__init__()
        self.num_heads = num_heads
        self.internal_similarity = internal_similarity
        tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
        tensor_2_dim = tensor_2_dim or tensor_1_dim
        tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
        if tensor_1_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_1_projected_dim, num_heads))
        if tensor_2_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_2_projected_dim, num_heads))
        self.tensor_1_projection = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
        self.tensor_2_projection = nn.Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.tensor_1_projection)
        torch.nn.init.xavier_uniform_(self.tensor_2_projection)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.tensor_1_projection)
        projected_tensor_2 = torch.matmul(tensor_2, self.tensor_2_projection)

        # Here we split the last dimension of the tensors from (..., projected_dim) to
        # (..., num_heads, projected_dim / num_heads), using tensor.view().
        last_dim_size = projected_tensor_1.size(-1) // self.num_heads
        new_shape = list(projected_tensor_1.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_1 = projected_tensor_1.view(*new_shape)
        last_dim_size = projected_tensor_2.size(-1) // self.num_heads
        new_shape = list(projected_tensor_2.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_2 = projected_tensor_2.view(*new_shape)

        # And then we pass this off to our internal similarity function. Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here. It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        return self.internal_similarity(split_tensor_1, split_tensor_2)