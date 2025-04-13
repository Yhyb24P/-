"""
引导滤波模块

提供边缘保持滤波功能，特别适合细胞边界增强。
"""

import cv2
import numpy as np


def guided_filter(image: np.ndarray,
                guide: np.ndarray = None,
                radius: int = 2,
                eps: float = 0.2) -> np.ndarray:
    """引导滤波

    边缘保持滤波器，可以在平滑图像的同时保持边缘。

    Args:
        image: 输入图像
        guide: 引导图像，如果为None则使用输入图像作为引导
        radius: 滤波半径
        eps: 正则化参数

    Returns:
        滤波后的图像
    """
    # 如果没有提供引导图像，使用输入图像作为引导
    if guide is None:
        guide = image

    # 转换为浮点数类型
    guide = guide.astype(np.float32)
    image = image.astype(np.float32)

    # 均值滤波
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_image = cv2.boxFilter(image, -1, (radius, radius))
    mean_guide_image = cv2.boxFilter(guide * image, -1, (radius, radius))

    # 协方差
    cov = mean_guide_image - mean_guide * mean_image

    # 方差
    var = cv2.boxFilter(guide * guide, -1, (radius, radius)) - mean_guide * mean_guide

    # 线性系数
    a = cov / (var + eps)
    b = mean_image - a * mean_guide

    # 对系数进行均值滤波
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # 最终输出
    output = mean_a * guide + mean_b

    # 转换回原始类型
    return np.clip(output, 0, 255).astype(np.uint8)


def fast_guided_filter(image: np.ndarray,
                      guide: np.ndarray = None,
                      radius: int = 2,
                      eps: float = 0.2,
                      scale: int = 4) -> np.ndarray:
    """快速引导滤波

    通过下采样加速的引导滤波实现

    Args:
        image: 输入图像
        guide: 引导图像，如果为None则使用输入图像作为引导
        radius: 滤波半径
        eps: 正则化参数
        scale: 下采样比例

    Returns:
        滤波后的图像
    """
    # 如果没有提供引导图像，使用输入图像作为引导
    if guide is None:
        guide = image

    # 下采样
    h, w = image.shape[:2]
    small_image = cv2.resize(image, (w // scale, h // scale))
    small_guide = cv2.resize(guide, (w // scale, h // scale))

    # 在低分辨率上应用引导滤波
    small_output = guided_filter(small_image, small_guide, radius // scale, eps)

    # 上采样回原始分辨率
    output = cv2.resize(small_output, (w, h))

    return output