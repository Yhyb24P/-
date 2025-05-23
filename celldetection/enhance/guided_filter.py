#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
引导滤波模块

提供边缘保持的引导滤波算法，用于细胞边界的增强保护。
"""

import cv2
import numpy as np

def guided_filter(guide, src, radius, eps):
    """引导滤波 - 边缘保持滤波器
    
    Args:
        guide: 引导图像
        src: 源图像
        radius: 滤波半径
        eps: 正则化参数
        
    Returns:
        滤波后的图像
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    # 均值滤波
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
    
    # 协方差
    cov = mean_guide_src - mean_guide * mean_src
    
    # 方差
    var = cv2.boxFilter(guide * guide, -1, (radius, radius)) - mean_guide * mean_guide
    
    # 线性系数
    a = cov / (var + eps)
    b = mean_src - a * mean_guide
    
    # 对系数进行均值滤波
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    
    # 最终输出
    output = mean_a * guide + mean_b
    
    return np.clip(output, 0, 255).astype(np.uint8)
