"""API 随手试验区：复制速查表的示例到这里，不影响练习答案。"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 在下面替换/追加你想尝试的 API 示例。
x = torch.arange(24).reshape(2, 3, 4)
print("shape:", x.shape)
print("transpose(1, 2):", x.transpose(1, 2).shape)
print("sum(dim=1):", x.sum(dim=1).shape)
print("x_looks_like",x)
print("x[0,2,3]",x[0,2,3])
y = x.reshape(1, 4, 6)
print("y.shape:", y.shape)
print("y_looks_like",y)
print("y_is_contiguous:",y.is_contiguous())
y = y.transpose(1, 2)
print("y_looks_like",y)
print("y.is_contiguous:", y.is_contiguous())
z = y.contiguous()
print("z_looks_like",z)
z = z.view(1, 4, 6)
print("z.is_contiguous:", z.is_contiguous())
print("z_looks_like",z)
coefficient = torch.tensor([
    [1., 2., 3.],   # 第 0 个样本，三个 token 各自的系数
    [4., 5., 6.],   # 第 1 个样本，三个 token 各自的系数
])
print("coefficient.shape:", coefficient.shape)
valid = torch.tensor([[True, False, True], [False, True, False]])
print("valid.unsqueeze(-1):", valid.unsqueeze(-1))
z = torch.ones(24).reshape(2,3,4)
print("z",z)
z = z.reshape(2,3,2,2)
print("z",z)
x = torch.ones(2,2,3)
y = torch.ones(2,2)
y[0,-1] = 0 
y[1,-1] = 0
print("y",y)
y = y.unsqueeze(-1)
print("y",y)
print("y*x",y*x)


