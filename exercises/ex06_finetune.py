"""06 冻结与训练。演示adapter位于冻结backbone之前的情况。"""
import torch
from torch import nn

class AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(4, 8)
        self.backbone = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.backbone(self.adapter(x)))

def freeze_backbone(model: AdapterModel):
    """只冻结backbone参数，清理backbone既有梯度；保留adapter/head可训练。"""
    raise NotImplementedError("TODO 06-A: 冻结参数而非截断计算图")

def train_step(model: nn.Module, optimizer, x, target,
               max_grad_norm: float = 1.0) -> float:
    """设为train、清梯度、MSE、backward、梯度裁剪、step；返回Python float。
    不在这里创建优化器，不累计上次梯度。
    """
    raise NotImplementedError("TODO 06-B: 一个训练step")

def gradient_report(model: nn.Module) -> dict[str, bool]:
    """参数名->本次是否有有限且非零梯度；grad为None返回False。"""
    raise NotImplementedError("TODO 06-C: 查参数梯度")
