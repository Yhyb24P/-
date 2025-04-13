"""
后处理模块

提供高级后处理功能，包括自适应NMS、密度感知NMS和细胞分裂检测等。
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union


def adaptive_nms(boxes: torch.Tensor,
                scores: torch.Tensor,
                iou_threshold: float = 0.5,
                density_aware: bool = True,
                min_threshold: float = 0.1,
                max_threshold: float = 0.5) -> torch.Tensor:
    """
    自适应NMS

    根据目标密度动态调整IoU阈值的NMS实现

    Args:
        boxes: 边界框坐标 [N, 4] 格式 (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        iou_threshold: 基准IoU阈值
        density_aware: 是否启用密度感知调整
        min_threshold: 最小IoU阈值
        max_threshold: 最大IoU阈值

    Returns:
        保留的边界框索引
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # 获取设备
    device = boxes.device

    # 对边界框格式进行转换，确保为xyxy格式
    if boxes.shape[1] > 4:
        boxes = boxes[:, :4]

    # 计算每个框的面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 按置信度降序排序
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        # 选择当前最高置信度的框
        i = order[0].item()
        keep.append(i)

        # 如果只剩下一个框，结束循环
        if order.numel() == 1:
            break

        # 计算当前框与剩余框的IoU
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        # 计算IoU
        iou = inter / (area[i] + area[order[1:]] - inter)

        # 密度感知调整
        if density_aware:
            # 根据局部密度调整阈值
            # 计算框i周围的框密度
            local_density = (iou > min_threshold).float().sum() / order.numel()

            # 根据密度调整阈值，密度越高，阈值越低
            adaptive_threshold = max_threshold - (max_threshold - min_threshold) * local_density

            # 应用自适应阈值
            inds = torch.where(iou <= adaptive_threshold)[0]
        else:
            # 使用固定阈值
            inds = torch.where(iou <= iou_threshold)[0]

        # 更新order
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long, device=device)


def soft_nms(boxes: torch.Tensor,
            scores: torch.Tensor,
            sigma: float = 0.5,
            score_threshold: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS

    不是硬性抑制重叠框，而是根据IoU减少其分数

    Args:
        boxes: 边界框坐标 [N, 4] 格式 (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        sigma: 高斯衰减参数
        score_threshold: 分数阈值，低于该阈值的框将被移除

    Returns:
        保留的边界框索引和更新后的分数
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device), torch.zeros(0, device=boxes.device)

    # 转为CPU进行处理
    if boxes.is_cuda:
        boxes_cpu = boxes.cpu()
        scores_cpu = scores.cpu()
    else:
        boxes_cpu = boxes
        scores_cpu = scores

    # 转为numpy数组
    boxes_np = boxes_cpu.numpy()
    scores_np = scores_cpu.numpy()

    # 计算每个框的面积
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 初始化结果
    keep = []
    updated_scores = []

    # 按置信度降序排序
    order = scores_np.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)
        updated_scores.append(scores_np[i])

        # 如果只剩下一个框，结束循环
        if order.size == 1:
            break

        # 计算当前框与剩余框的IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # 计算IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 使用高斯衰减函数更新分数
        weight = np.exp(-(iou * iou) / sigma)
        scores_np[order[1:]] *= weight

        # 移除分数低于阈值的框
        remain_inds = np.where(scores_np[order[1:]] > score_threshold)[0]
        order = order[remain_inds + 1]

    # 转回tensor
    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    updated_scores_tensor = torch.tensor(updated_scores, device=scores.device)

    return keep_tensor, updated_scores_tensor


def detect_budding_cells(image: np.ndarray,
                        boxes: np.ndarray,
                        threshold: float = 0.15) -> np.ndarray:
    """
    检测出芽（分裂中）的酵母细胞

    通过形态学特征识别出芽的酵母细胞

    Args:
        image: 输入图像
        boxes: 边界框坐标 [N, 4] 格式 (x1, y1, x2, y2)
        threshold: 判断细胞出芽的形状阈值

    Returns:
        每个细胞的标签，1表示出芽细胞，0表示普通细胞
    """
    if len(boxes) == 0:
        return np.array([])

    # 获取图像尺寸
    h, w = image.shape[:2]

    # 转为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 初始化结果
    results = np.zeros(len(boxes), dtype=np.int32)

    # 分析每个检测框
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])

        # 确保边界在图像内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 提取ROI
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # 二值化
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # 选择最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area == 0:
            continue

        # 计算周长
        perimeter = cv2.arcLength(max_contour, True)

        # 计算圆度 (4π * 面积 / 周长^2)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 1

        # 计算椭圆拟合
        if len(max_contour) >= 5:  # 需要至少5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(max_contour)
            (_, _), (ma, mi), _ = ellipse

            # 计算长短轴比
            if mi > 0:
                aspect_ratio = ma / mi
            else:
                aspect_ratio = 1
        else:
            aspect_ratio = 1

        # 计算凸度
        hull = cv2.convexHull(max_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            convexity = area / hull_area
        else:
            convexity = 1

        # 出芽细胞通常形状不规则：圆度低、长短轴比大、凸度低
        # 使用多特征组合判断
        budding_score = (1 - circularity) * 0.4 + (aspect_ratio - 1) * 0.4 + (1 - convexity) * 0.2

        # 根据阈值判断
        if budding_score > threshold:
            results[i] = 1  # 标记为出芽细胞

    return results