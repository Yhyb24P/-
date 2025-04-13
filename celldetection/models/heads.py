#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测头部实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv10Head(nn.Module):
    """YOLO v10检测头"""
    
    def __init__(self, num_classes=4, anchors=None):
        """
        初始化
        
        参数:
            num_classes: 类别数量
            anchors: 锚点配置
        """
        super().__init__()
        
        if anchors is None:
            # 默认锚点配置，针对酵母细胞检测优化
            self.anchors = [
                [[10, 10], [20, 20], [30, 30]],  # 小尺度特征图 (P3)
                [[30, 30], [50, 50], [70, 70]],  # 中尺度特征图 (P4)
                [[70, 70], [100, 100], [150, 150]]  # 大尺度特征图 (P5)
            ]
        else:
            self.anchors = anchors
        
        self.num_classes = num_classes
        
        # 每个尺度的特征图的检测头
        self.head_p3 = self._build_head(256)  # P3检测头
        self.head_p4 = self._build_head(512)  # P4检测头
        self.head_p5 = self._build_head(1024)  # P5检测头
    
    def _build_head(self, in_channels):
        """
        构建单个尺度的检测头
        
        参数:
            in_channels: 输入通道数
            
        返回:
            检测头模块
        """
        # 每个锚点预测 (5 + num_classes) 个值:
        # 4个边界框坐标 + 1个对象性 + num_classes个类别概率
        out_channels = 3 * (5 + self.num_classes)
        
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, features):
        """
        前向传播
        
        参数:
            features: 特征列表 [p3, p4, p5]
            
        返回:
            检测结果列表
        """
        p3, p4, p5 = features
        
        # 应用检测头
        pred_p3 = self.head_p3(p3)
        pred_p4 = self.head_p4(p4)
        pred_p5 = self.head_p5(p5)
        
        # 重塑预测形状以解析预测
        batch_size = p3.shape[0]
        
        # 处理P3预测 (小目标)
        h3, w3 = p3.shape[2:4]
        pred_p3 = pred_p3.view(batch_size, 3, 5 + self.num_classes, h3, w3)
        pred_p3 = pred_p3.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 5+C]
        
        # 处理P4预测 (中目标)
        h4, w4 = p4.shape[2:4]
        pred_p4 = pred_p4.view(batch_size, 3, 5 + self.num_classes, h4, w4)
        pred_p4 = pred_p4.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 5+C]
        
        # 处理P5预测 (大目标)
        h5, w5 = p5.shape[2:4]
        pred_p5 = pred_p5.view(batch_size, 3, 5 + self.num_classes, h5, w5)
        pred_p5 = pred_p5.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 5+C]
        
        return [pred_p3, pred_p4, pred_p5] 