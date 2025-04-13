"""
回归损失函数模块

提供常用的回归损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    """均方误差损失
    
    包装了 torch.nn.MSELoss
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.MSELoss(**kwargs)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            input: 预测结果
            target: 目标值
            
        Returns:
            损失值
        """
        return self.loss(input, target)

class SmoothL1Loss(nn.Module):
    """平滑L1损失
    
    包装了 torch.nn.SmoothL1Loss
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.SmoothL1Loss(**kwargs)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            input: 预测结果
            target: 目标值
            
        Returns:
            损失值
        """
        return self.loss(input, target)

# 可以根据需要添加其他回归损失函数，例如 IoU Loss, GIoU Loss 等
# class IoULoss(nn.Module):
#     ... 