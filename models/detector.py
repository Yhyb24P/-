#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测器模型

提供用于检测显微镜图像中酵母细胞的主要模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.convs import DepthwiseSeparableConv
from models.neck.fpn import LightweightFeaturePyramid
from models.attention.cbam import LightweightCBAM

# 定义检测器模型
class YeastCellDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(YeastCellDetector, self).__init__()
        # 第一层：标准卷积提取基础特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # 第二层：深度可分离卷积
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2)
        )
        
        # 第三层：深度可分离卷积
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2)
        )
        
        # 第四层：深度可分离卷积
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2)
        )
        
        # 特征金字塔网络 (FPN) - 使用轻量级设计
        self.fpn = LightweightFeaturePyramid(
            in_channels_list=[64, 128, 256],
            out_channels=128
        )
        
        # 轻量级注意力模块
        self.cbam = LightweightCBAM(128, reduction_ratio=8)
        
        # 检测头 - 使用深度可分离卷积
        self.detector = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.Conv2d(128, 5 + num_classes, kernel_size=1)  # x, y, w, h, obj, classes
        )
    
    def forward(self, x):
        # 提取多尺度特征
        features = []
        
        x1 = self.conv1(x)
        
        x2 = self.conv2(x1)
        features.append(x2)  # 第一层FPN特征
        
        x3 = self.conv3(x2)
        features.append(x3)  # 第二层FPN特征
        
        x4 = self.conv4(x3)
        features.append(x4)  # 第三层FPN特征
        
        # 应用FPN融合多尺度特征
        fpn_features = self.fpn(features)
        
        # 选择最适合的特征图用于检测
        detection_features = fpn_features[1]  # 中间层特征通常最适合细胞检测
        
        # 应用注意力机制
        detection_features = self.cbam(detection_features)
        
        # 预测
        output = self.detector(detection_features)
        
        # 重塑输出为 [B, H*W, 5+num_classes]
        batch_size, channels, height, width = output.shape
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, height * width, channels)
        
        return output

# 兼容原SimpleCellDetector名称
SimpleCellDetector = YeastCellDetector 