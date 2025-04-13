"""
检测指标计算模块

提供目标检测相关的评估指标计算功能
"""

import numpy as np
from typing import List, Dict, Any

def calculate_metrics(targets: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算检测指标
    
    Args:
        targets: 目标标注列表，每个元素包含bbox和class_id
        predictions: 预测结果列表，每个元素包含bbox、class_id和confidence
        
    Returns:
        包含各项指标的字典
    """
    if not targets or not predictions:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mAP': 0.0
        }
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(targets), len(predictions)))
    for i, target in enumerate(targets):
        for j, pred in enumerate(predictions):
            iou_matrix[i, j] = calculate_iou(target['bbox'], pred['bbox'])
    
    # 设置IoU阈值
    iou_threshold = 0.5
    
    # 计算TP、FP、FN
    tp = 0
    fp = 0
    fn = 0
    
    # 对每个目标找到最佳匹配的预测
    matched = set()
    for i in range(len(targets)):
        best_iou = 0
        best_j = -1
        
        for j in range(len(predictions)):
            if j in matched:
                continue
                
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        
        if best_iou >= iou_threshold and best_j != -1:
            tp += 1
            matched.add(best_j)
        else:
            fn += 1
    
    fp = len(predictions) - len(matched)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mAP': precision  # 简化版本，实际应该计算不同IoU阈值下的AP
    } 