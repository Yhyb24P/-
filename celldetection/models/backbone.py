#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测骨干网络
"""

import torch
import torch.nn as nn

class YeastBackbone(nn.Module):
    """酵母细胞检测专用骨干网络"""
    
    def __init__(self, in_channels=3):
        """
        初始化
        
        参数:
            in_channels: 输入通道数
        """
        super().__init__()
        
        # 定义主干网络各阶段
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU()
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [B, 3, H, W]
            
        返回:
            特征映射列表 [p3, p4, p5]
        """
        # 阶段1
        x1 = self.stage1(x)
        
        # 阶段2
        x2 = self.stage2(x1)
        
        # 阶段3
        x3 = self.stage3(x2)
        
        # 返回多尺度特征
        return [x1, x2, x3]  # [p3, p4, p5] 