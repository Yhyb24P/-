#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class YeastAttention(nn.Module):
    """酵母细胞专用注意力机制"""
    
    def __init__(self, in_channels):
        """
        初始化
        
        参数:
            in_channels: 输入通道数
        """
        super().__init__()
        
        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 形态注意力 - 使用深度可分离卷积提取细胞形态特征
        self.morphology_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, C, H, W]
            
        返回:
            注意力加权特征
        """
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa
        
        # 形态注意力 - 专门针对细胞形态特征
        ma = self.morphology_attention(x_sa)
        x_ma = x_sa * ma
        
        return x_ma 