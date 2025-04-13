#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CBAM注意力机制

提供轻量级的Convolutional Block Attention Module (CBAM)实现。
包括通道注意力和空间注意力机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 轻量级CBAM注意力机制
class LightweightCBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(LightweightCBAM, self).__init__()
        # 通道注意力
        self.channel_attention = LightweightChannelAttention(channels, reduction_ratio)
        # 空间注意力 - 简化版本
        self.spatial_attention = LightweightSpatialAttention()
    
    def forward(self, x):
        # 应用通道注意力
        x = self.channel_attention(x) * x
        # 应用空间注意力
        x = self.spatial_attention(x) * x
        return x

# 轻量级通道注意力模块
class LightweightChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(LightweightChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 仅使用平均池化，省略最大池化以减少计算量
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out)

# 轻量级空间注意力模块
class LightweightSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(LightweightSpatialAttention, self).__init__()
        # 使用单个通道卷积代替标准卷积
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 仅使用平均池化在通道维度上
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = self.conv(avg_out)
        return self.sigmoid(out) 