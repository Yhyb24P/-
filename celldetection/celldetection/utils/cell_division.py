"""
细胞分裂检测模块

提供检测和分析细胞分裂状态的功能，特别是酵母细胞的出芽状态。
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union, Any


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


def analyze_cell_cycle(image: np.ndarray,
                      boxes: np.ndarray,
                      budding_threshold: float = 0.15) -> Dict[str, Any]:
    """
    分析细胞周期状态
    
    根据形态学特征分析细胞周期状态
    
    Args:
        image: 输入图像
        boxes: 边界框坐标 [N, 4] 格式 (x1, y1, x2, y2)
        budding_threshold: 判断细胞出芽的形状阈值
        
    Returns:
        细胞周期分析结果，包括各阶段细胞数量和比例
    """
    if len(boxes) == 0:
        return {
            'total_cells': 0,
            'budding_cells': 0,
            'non_budding_cells': 0,
            'budding_ratio': 0.0,
            'cell_states': []
        }
    
    # 检测出芽细胞
    budding_labels = detect_budding_cells(image, boxes, budding_threshold)
    
    # 统计结果
    total_cells = len(boxes)
    budding_cells = np.sum(budding_labels)
    non_budding_cells = total_cells - budding_cells
    budding_ratio = budding_cells / total_cells if total_cells > 0 else 0.0
    
    # 细胞状态列表
    cell_states = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # 计算细胞大小
        cell_size = (x2 - x1) * (y2 - y1)
        
        # 判断细胞状态
        if budding_labels[i] == 1:
            state = 'budding'
        else:
            state = 'non_budding'
            
        cell_states.append({
            'box': box.tolist(),
            'state': state,
            'size': cell_size
        })
    
    return {
        'total_cells': total_cells,
        'budding_cells': int(budding_cells),
        'non_budding_cells': int(non_budding_cells),
        'budding_ratio': float(budding_ratio),
        'cell_states': cell_states
    }


def detect_cell_pairs(boxes: np.ndarray, 
                     iou_threshold: float = 0.2,
                     distance_threshold: float = 30) -> List[Tuple[int, int]]:
    """
    检测可能的细胞对（母细胞和子细胞）
    
    根据位置关系识别可能的母细胞和子细胞对
    
    Args:
        boxes: 边界框坐标 [N, 4] 格式 (x1, y1, x2, y2)
        iou_threshold: IoU阈值，用于判断细胞是否相邻
        distance_threshold: 距离阈值，用于判断细胞是否相邻
        
    Returns:
        可能的细胞对索引列表 [(i, j), ...]，其中i和j是boxes中的索引
    """
    if len(boxes) < 2:
        return []
    
    # 计算所有细胞对之间的IoU
    def calculate_iou(box1, box2):
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        inter_area = w * h
        
        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    # 计算所有细胞对之间的中心点距离
    def calculate_distance(box1, box2):
        # 计算中心点
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        
        # 计算欧氏距离
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance
    
    # 计算所有细胞对之间的大小比例
    def calculate_size_ratio(box1, box2):
        # 计算面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算大小比例 (大/小)
        ratio = max(area1, area2) / min(area1, area2) if min(area1, area2) > 0 else float('inf')
        
        return ratio
    
    # 查找可能的细胞对
    cell_pairs = []
    
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            box1 = boxes[i]
            box2 = boxes[j]
            
            # 计算IoU
            iou = calculate_iou(box1, box2)
            
            # 计算距离
            distance = calculate_distance(box1, box2)
            
            # 计算大小比例
            size_ratio = calculate_size_ratio(box1, box2)
            
            # 判断是否为可能的细胞对
            # 1. IoU适中（不太高也不太低）
            # 2. 距离适中
            # 3. 大小比例合理（母细胞通常比子细胞大）
            if (iou > iou_threshold and 
                distance < distance_threshold and 
                size_ratio < 3.0):
                
                # 确定母细胞和子细胞
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                if area1 >= area2:
                    cell_pairs.append((i, j))  # i是母细胞，j是子细胞
                else:
                    cell_pairs.append((j, i))  # j是母细胞，i是子细胞
    
    return cell_pairs


def visualize_cell_division(image: np.ndarray,
                           boxes: np.ndarray,
                           budding_labels: Optional[np.ndarray] = None,
                           cell_pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    可视化细胞分裂状态
    
    在图像上标注细胞分裂状态
    
    Args:
        image: 输入图像
        boxes: 边界框坐标 [N, 4] 格式 (x1, y1, x2, y2)
        budding_labels: 出芽标签，1表示出芽细胞，0表示普通细胞
        cell_pairs: 细胞对索引列表 [(i, j), ...]
        
    Returns:
        可视化结果图像
    """
    if len(boxes) == 0:
        return image.copy()
    
    # 复制图像
    result = image.copy()
    
    # 如果没有提供出芽标签，则检测出芽细胞
    if budding_labels is None:
        budding_labels = detect_budding_cells(image, boxes)
    
    # 绘制所有细胞
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # 根据细胞状态选择颜色
        if budding_labels[i] == 1:
            color = (0, 0, 255)  # 红色表示出芽细胞
        else:
            color = (0, 255, 0)  # 绿色表示普通细胞
            
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 添加标签
        label = 'Budding' if budding_labels[i] == 1 else 'Normal'
        cv2.putText(result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 绘制细胞对连线
    if cell_pairs is not None:
        for mother_idx, daughter_idx in cell_pairs:
            # 获取母细胞和子细胞的中心点
            mother_box = boxes[mother_idx]
            daughter_box = boxes[daughter_idx]
            
            mother_center = (int((mother_box[0] + mother_box[2]) / 2), 
                            int((mother_box[1] + mother_box[3]) / 2))
            
            daughter_center = (int((daughter_box[0] + daughter_box[2]) / 2), 
                              int((daughter_box[1] + daughter_box[3]) / 2))
            
            # 绘制连线
            cv2.line(result, mother_center, daughter_center, (255, 0, 255), 2)
            
            # 添加标签
            cv2.putText(result, 'M', mother_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(result, 'D', daughter_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return result
