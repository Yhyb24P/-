"""
CLAHE增强模块

提供对比度受限的自适应直方图均衡化功能。
特别适合显微镜图像的对比度增强。
"""

import cv2
import numpy as np
from typing import Tuple


def apply_clahe(image: np.ndarray,
               clip_limit: float = 2.0,
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """应用CLAHE（对比度受限的自适应直方图均衡化）

    Args:
        image: 输入图像
        clip_limit: 对比度限制
        tile_grid_size: 网格大小

    Returns:
        增强后的图像
    """
    # 转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 应用CLAHE到L通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)

    # 合并通道
    lab = cv2.merge((l, a, b))

    # 转换回RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


def adaptive_clahe(image: np.ndarray) -> np.ndarray:
    """自适应CLAHE增强

    根据图像特性自动选择CLAHE参数

    Args:
        image: 输入图像

    Returns:
        增强后的图像
    """
    # 转换为灰度图以分析图像特性
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    std = np.std(gray)

    # 根据图像特性调整参数
    clip_limit = max(1.0, min(4.0, 5.0 - std / 50))  # 标准差小的图像需要更大的clip limit

    # 选择网格大小
    if std < 30:  # 低对比度图像
        tile_grid_size = (4, 4)  # 使用更小的网格增强局部对比度
    else:
        tile_grid_size = (8, 8)  # 标准网格大小

    # 应用CLAHE
    return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)