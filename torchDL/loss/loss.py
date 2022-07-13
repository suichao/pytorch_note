import numpy as np
import sys


def has_modules(modulename):
    has_flag = False
    if modulename in sys.modules:
        has_flag = True
    return has_flag


class LossBase(object):
    """
    Loss基类
    """
    def __init__(self):
        self.input_type = None

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def compatible_input(self, inputs):
        if has_modules("torch"):
            import torch
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.numpy()
                self.input_type = "torch"
        if has_modules("tensorflow"):
            import tensorflow
            if isinstance(inputs, tensorflow.Tensor):
                inputs = inputs.numpy()
                self.input_type = "tensorflow"
        if isinstance(inputs, list):
            inputs = np.array(inputs)
            self.input_type = "list"
        return inputs

    def convert(self, inputs):
        if self.input_type == 'torch':
            import torch
            inputs = torch.tensor(inputs)
        elif self.input_type == 'tensorflow':
            import tensorflow
            inputs = tensorflow.convert_to_tensor(inputs)
        elif self.input_type == 'list':
            inputs = inputs.tolist()
        return inputs

    def call(self, predict, label):
        predict = self.compatible_input(predict)
        label = self.compatible_input(label)
        loss_v = self.compute_loss(predict, label)
        loss_v = self.convert(loss_v)
        self.input_type = None
        return loss_v

    def compute_loss(self, predict, label):
        raise NotImplementedError


class KlLoss(LossBase):
    """
    相对熵(relative entropy)又称为KL散度（Kullback-Leibler divergence）
    用于衡量一个分布相对于另一个分布的差异性，差异越小，kl散度为0，否则为1。
    模型训练使用kl散度作为损失，使得一个分布逼近另一个分布；
    熵的定义式：

    p(xi)为近似分布， q(xi)为真实分布
    Dkl(p||q) = sum(q(xi)ln(q(xi)/p(xi)))

    将定义式张开，从熵的角度看kl散度：
    KL散度=交叉熵-熵
    对于给定训练集，熵是已知的，那么求取KL散度等价于求取交叉熵，因此交叉熵才被用作代价函数

    import scipy.stats
    # 随机生成两个离散型分布
    x = [np.random.randint(1, 11) for _ in range(10)]
    px = x / np.sum(x)
    print(px)
    y = [np.random.randint(1, 11) for _ in range(10)]
    py = y / np.sum(y)
    print(py)

    KL = scipy.stats.entropy(px, py)
    print(KL)

    KL = 0.0
    for i in range(10):
        KL += px[i] * np.log(px[i] / py[i])
    print(KL)
    """
    def compute_loss(self, predict, label):
        predict = predict.flatten()
        label = label.flatten()
        kl = sum(label * np.log(label / predict))
        return kl


class CrossEntropyLoss(LossBase):
    """
    交叉熵
    p(x)常用于描述样本的真实分布，例如[1,0,0,0]表示样本属于第一类，而q(x)则常常用于表示预测的分布，例如[0.7,0.1,0.1,0.1]
    """
    def compute_loss(self, predict, label):
        predict = predict.flatten()
        label = label.flatten()
        return - sum(label * np.log(predict))


class BinaryCrossEntropyLoss(LossBase):
    """
    二元交叉熵
    p(x)常用于描述样本的真实分布，例如[1,0,0,0]表示样本属于第一类，而q(x)则常常用于表示预测的分布，例如[0.7,0.1,0.1,0.1]
    """
    def compute_loss(self, predict, label):
        predict = predict.flatten()
        label = label.flatten()
        return sum((1 - label) * np.log((1 - predict)) - label * np.log(predict))

