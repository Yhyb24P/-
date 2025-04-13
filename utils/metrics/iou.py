"""
IoU计算模块

提供边界框IoU(Intersection over Union)的计算功能
"""

import numpy as np
from typing import List, Union

def calculate_iou(box1: Union[List[float], np.ndarray], 
                 box2: Union[List[float], np.ndarray]) -> float:
    """计算两个边界框的IoU
    
    Args:
        box1: 第一个边界框 [x, y, w, h]
        box2: 第二个边界框 [x, y, w, h]
        
    Returns:
        IoU值
    """
    # 转换为numpy数组
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # 计算边界框的坐标
    x1 = box1[0]
    y1 = box1[1]
    w1 = box1[2]
    h1 = box1[3]
    
    x2 = box2[0]
    y2 = box2[1]
    w2 = box2[2]
    h2 = box2[3]
    
    # 计算交集区域的坐标
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # 计算交集面积
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个边界框的面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # 计算IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    
    return iou 