"""
后处理功能包

本包提供高级后处理功能，包括自适应NMS、细胞分裂检测等。
"""

# 导出自适应NMS相关函数
from .adaptive_nms import (
    adaptive_nms,
    soft_nms,
    connected_component_analysis,
    density_based_nms,
    multi_threshold_nms
)

# 其他功能将在各自的模块中实现
# 如：细胞分裂检测等

__all__ = [
    'adaptive_nms',
    'soft_nms',
    'connected_component_analysis',
    'density_based_nms', 
    'multi_threshold_nms'
] 