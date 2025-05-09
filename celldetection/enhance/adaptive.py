#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自适应图像增强模块

提供自适应图像增强算法，专门用于显微镜下的酵母细胞图像。
"""

import cv2
import numpy as np
from typing import Tuple

# 导入其他增强模块
from .guided_filter import guided_filter
from .small_cell import enhance_small_cells

class AdaptiveEnhance:
    """自适应图像增强类"""
    
    def __init__(self, contrast_limit=2.0, brightness_offset=10):
        """
        初始化
        
        参数:
            contrast_limit: 对比度增强限制
            brightness_offset: 亮度补偿值
        """
        self.contrast_limit = contrast_limit
        self.brightness_offset = brightness_offset
    
    def enhance(self, image):
        """
        增强图像
        
        参数:
            image: 输入图像
            
        返回:
            增强后的图像
        """
        # 灰度图转换
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 估计噪声水平
        noise_level = self.estimate_noise(gray)
        
        # 根据噪声水平调整参数
        alpha = max(1.0, min(self.contrast_limit, 2.0 - noise_level / 10.0))
        beta = self.brightness_offset * (1.0 - noise_level / 20.0)
        
        # 自适应对比度增强
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 如果是彩色图像，进行颜色平衡
        if len(image.shape) == 3:
            # 分离通道
            b, g, r = cv2.split(enhanced)
            
            # 通道均衡化
            b_eq = cv2.equalizeHist(b)
            g_eq = cv2.equalizeHist(g)
            r_eq = cv2.equalizeHist(r)
            
            # 合并通道
            enhanced = cv2.merge([b_eq, g_eq, r_eq])
            
            # 使用自适应混合系数
            blend_ratio = min(0.7, max(0.3, 0.5 - noise_level / 30.0))
            enhanced = cv2.addWeighted(enhanced, blend_ratio, image, 1 - blend_ratio, 0)
        
        return enhanced
    
    def estimate_noise(self, image):
        """
        估计图像噪声水平
        
        参数:
            image: 输入灰度图像
            
        返回:
            估计的噪声水平
        """
        # 拉普拉斯操作
        lap = cv2.Laplacian(image, cv2.CV_64F)
        
        # 计算标准差作为噪声估计
        noise_sigma = np.std(lap)
        
        return noise_sigma

def enhance_microscopy_image(image: np.ndarray) -> np.ndarray:
    """增强显微镜图像
    
    Args:
        image: 输入图像
        
    Returns:
        增强后的图像
    """
    # 转换为灰度图以分析图像特性
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    
    # 保存原始图像以便后续处理
    enhanced = image.copy()
    
    # 1. 自适应CLAHE增强 - 基于图像统计特性动态调整参数
    clahe_clip_limit = max(1.0, min(4.0, 5.0 - std / 50))  # 根据标准差动态调整clip limit
    grid_size = (8, 8)
    if std < 30:  # 低对比度图像使用更小的网格
        grid_size = (4, 4)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=grid_size)
    
    # 转换到LAB颜色空间 - 只对L通道应用CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 2. 引导滤波 - 边缘保持滤波，特别适合细胞边界增强
    radius = 2  # 小半径更适合酵母细胞这类小目标
    eps = 0.2   # 较小的eps值保留更多细节
    
    for i in range(3):
        enhanced[:, :, i] = guided_filter(enhanced[:, :, i], enhanced[:, :, i], radius, eps)
    
    # 3. 适应性锐化 - 根据图像特性调整锐化参数
    kernel_strength = 5 + int(10 * (1 - std / 255))  # 低对比度图像使用更强的锐化
    kernel_center = 1 + kernel_strength
    
    kernel = np.array([[-1, -1, -1],
                       [-1, kernel_center, -1],
                       [-1, -1, -1]])
    kernel = kernel / kernel.sum()  # 归一化以保持亮度
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. 小目标增强 - 专门针对酵母细胞这类小目标
    enhanced = enhance_small_cells(enhanced, cell_size_threshold=0.005)
    
    # 5. 自适应亮度调整
    if mean < 100:
        # 创建亮度掩码 - 只增强暗区域
        bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        bright_mask = cv2.dilate(bright_mask, np.ones((3,3), np.uint8))
        bright_mask = cv2.GaussianBlur(bright_mask, (5,5), 0)
        
        # 增强暗区域
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        # 只调整暗区域
        v_adjusted = np.where(bright_mask == 0, 
                             np.clip(v * 1.3, 0, 255).astype(np.uint8), 
                             v)
        hsv_adjusted = cv2.merge([h, s, v_adjusted])
        enhanced_dark = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
        
        # 根据掩码融合结果
        bright_mask_norm = bright_mask / 255.0
        enhanced = enhanced_dark * (1 - bright_mask_norm[:,:,np.newaxis]) + enhanced * bright_mask_norm[:,:,np.newaxis]
        enhanced = enhanced.astype(np.uint8)
    
    # 6. 最终应用轻微去噪 - 使用非局部均值去噪保留细节
    noise_level = estimate_noise(gray)
    if noise_level > 5:
        h_param = int(max(3, min(10, noise_level / 2)))
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h_param, h_param, 7, 21)
    
    return enhanced
