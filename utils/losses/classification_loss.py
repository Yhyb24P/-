"""
分类损失函数模块

提供常用的分类损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """交叉熵损失
    
    包装了 torch.nn.CrossEntropyLoss
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            input: 预测结果，形状 [N, C]
            target: 目标标签，形状 [N]
            
        Returns:
            损失值
        """
        return self.loss(input, target)

# 可以根据需要添加其他分类损失函数，例如 Focal Loss
# class FocalLoss(nn.Module):
#     ... 