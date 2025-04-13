"""
自适应图像增强模块

提供针对显微镜下酵母细胞图像的自适应增强功能。
根据图像特性自动选择最佳的增强方法。
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def enhance_microscopy_image(image: np.ndarray) -> np.ndarray:
    """增强显微镜图像

    根据图像特性自动选择最佳的增强方法

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

    # 3. 亮度调整 - 对于低亮度图像
    if mean < 100:
        # 使用Gamma校正调整亮度
        gamma = 1.5  # gamma > 1 使暗区更亮
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype('uint8')
        enhanced = cv2.LUT(enhanced, table)

    # 4. 针对高噪声图像的处理
    noise_level = estimate_noise(gray)
    if noise_level > 10:
        # 使用双边滤波保留边缘的同时去除噪声
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

    # 5. 针对模糊图像的锐化
    if estimate_blur(gray) > 0.5:
        # 使用锐化滤波增强边缘
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 6. 最终应用轻微去噪 - 使用非局部均值去噪保留细节
    if noise_level > 5:
        h_param = int(max(3, min(10, noise_level / 2)))
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h_param, h_param, 7, 21)

    return enhanced


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
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


def estimate_noise(image: np.ndarray) -> float:
    """估计图像噪声水平

    Args:
        image: 输入灰度图像

    Returns:
        噪声水平估计值
    """
    # 使用拉普拉斯算子估计噪声
    H, W = image.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image.astype(np.float32), -1, np.array(M)))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma


def estimate_blur(image: np.ndarray) -> float:
    """估计图像模糊程度

    Args:
        image: 输入灰度图像

    Returns:
        模糊程度估计值，范围[0,1]，值越大表示越模糊
    """
    # 使用拉普拉斯算子计算模糊度
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap_std = np.std(lap)

    # 标准化到[0,1]范围
    blur_index = 1.0 - min(lap_std / 50.0, 1.0)  # 假设最大拉普拉斯标准差为50

    return blur_index