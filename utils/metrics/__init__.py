"""
指标计算模块

提供各种评估指标的计算功能，包括：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1 Score)
- 平均精度 (mAP)
- IoU (Intersection over Union)
"""

from .detection_metrics import calculate_metrics
from .iou import calculate_iou
from .map import calculate_map

__all__ = [
    'calculate_metrics',
    'calculate_iou',
    'calculate_map'
] 