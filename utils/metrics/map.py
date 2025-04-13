"""
mAP计算模块

提供平均精度(mAP)的计算功能
"""

import numpy as np
from typing import List, Dict, Any
from .iou import calculate_iou

def calculate_map(targets: List[Dict[str, Any]], 
                 predictions: List[Dict[str, Any]], 
                 iou_thresholds: List[float] = None) -> float:
    """计算平均精度(mAP)
    
    Args:
        targets: 目标标注列表
        predictions: 预测结果列表
        iou_thresholds: IoU阈值列表，默认为[0.5, 0.75, 0.9]
        
    Returns:
        mAP值
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.9]
        
    if not targets or not predictions:
        return 0.0
        
    aps = []
    for iou_threshold in iou_thresholds:
        ap = calculate_ap_at_iou(targets, predictions, iou_threshold)
        aps.append(ap)
        
    return np.mean(aps)

def calculate_ap_at_iou(targets: List[Dict[str, Any]], 
                       predictions: List[Dict[str, Any]], 
                       iou_threshold: float) -> float:
    """计算特定IoU阈值下的AP
    
    Args:
        targets: 目标标注列表
        predictions: 预测结果列表
        iou_threshold: IoU阈值
        
    Returns:
        AP值
    """
    # 按置信度排序预测结果
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # 记录已匹配的目标
    matched_targets = set()
    
    # 对每个预测结果
    for i, pred in enumerate(predictions):
        best_iou = 0
        best_target_idx = -1
        
        # 寻找最佳匹配的目标
        for j, target in enumerate(targets):
            if j in matched_targets:
                continue
                
            iou = calculate_iou(pred['bbox'], target['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_target_idx = j
                
        # 根据IoU阈值判断是否为TP
        if best_iou >= iou_threshold and best_target_idx != -1:
            tp[i] = 1
            matched_targets.add(best_target_idx)
        else:
            fp[i] = 1
            
    # 计算累积值
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # 计算精确率和召回率
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(targets)
    
    # 计算AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask])
            
    return ap / 11.0  # 11个点平均 