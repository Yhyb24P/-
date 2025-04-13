"""
小细胞增强模块

提供针对小细胞目标的特殊增强功能。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def enhance_small_cells(image: np.ndarray, cell_size_threshold: float = 0.005) -> np.ndarray:
    """增强小细胞目标

    检测并增强图像中的小细胞目标

    Args:
        image: 输入图像
        cell_size_threshold: 小细胞的相对大小阈值

    Returns:
        增强后的图像
    """
    # 图像尺寸
    h, w = image.shape[:2]
    min_dim = min(h, w)

    # 计算小细胞的像素尺寸阈值
    cell_pixel_threshold = int(min_dim * cell_size_threshold)

    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 自适应阈值处理
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制原始图像以进行增强
    enhanced_img = image.copy()

    # 处理每个轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 只处理小细胞
        if area < cell_pixel_threshold * cell_pixel_threshold * np.pi:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 扩大ROI区域以包含周围上下文
            padding = 2
            min_x = max(0, x - padding)
            min_y = max(0, y - padding)
            max_x = min(image.shape[1] - 1, x + w + padding)
            max_y = min(image.shape[0] - 1, y + h + padding)

            # 确保ROI有效
            if min_y < max_y and min_x < max_x:
                roi = enhanced_img[min_y:max_y+1, min_x:max_x+1].copy()

                # 应用局部增强
                roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(roi_lab)

                # 增强亮度通道
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
                l_enhanced = clahe.apply(l)

                # 锐化
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                l_enhanced = cv2.filter2D(l_enhanced, -1, kernel)

                # 合并通道
                roi_lab_enhanced = cv2.merge([l_enhanced, a, b])
                roi_enhanced = cv2.cvtColor(roi_lab_enhanced, cv2.COLOR_LAB2RGB)

                # 将增强的ROI放回原图
                enhanced_img[min_y:max_y+1, min_x:max_x+1] = roi_enhanced

    return enhanced_img


def detect_small_cells(image: np.ndarray,
                      min_area: int = 10,
                      max_area: int = 200) -> List[Tuple[int, int, int, int]]:
    """检测小细胞

    返回图像中小细胞的边界框

    Args:
        image: 输入图像
        min_area: 最小细胞面积
        max_area: 最大细胞面积

    Returns:
        小细胞边界框列表 [(x, y, w, h), ...]
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓
    small_cells = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            small_cells.append((x, y, w, h))

    return small_cells