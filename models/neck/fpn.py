#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征金字塔网络(FPN)模块

提供轻量级特征金字塔网络实现，用于多尺度特征融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 引入深度可分离卷积
from models.backbone.convs import DepthwiseSeparableConv

# 轻量级特征金字塔网络模块
class LightweightFeaturePyramid(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(LightweightFeaturePyramid, self).__init__()
        
        # 横向连接层 - 1x1卷积
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels in in_channels_list
        ])
        
        # 特征增强层 - 深度可分离卷积
        self.fpn_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])
    
    def forward(self, x):
        # 横向连接
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, x)]
        
        # 自顶向下路径和融合
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样
            laterals[i-1] += nn.functional.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
        
        # 最终输出
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        return outputs 