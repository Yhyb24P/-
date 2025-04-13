#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测主包

为显微镜下的酵母细胞图像提供检测、计数、分类和增强功能。
"""

__version__ = '0.1.0'
__author__ = '酵母细胞检测团队'

# 导入增强模块以便直接从包中使用
from celldetection.enhance import (
    enhance_microscopy_image,
    enhance_small_cells,
    guided_filter,
    apply_clahe,
    adaptive_clahe
)

# 尝试导入模型组件
try:
    from models.detector import YeastCellDetector
    from data.datasets.cell_dataset import CellDataset
    from data.utils.loss import SimpleCellLoss
except ImportError:
    print("警告: 未找到模型组件，部分功能可能无法使用。")

# 用于识别模块是否已正确安装的变量
is_installed = True

def version():
    """返回版本信息"""
    return __version__

def get_config_dir():
    """返回配置目录路径"""
    import os
    from pathlib import Path
    
    # 默认配置目录
    return os.path.join(Path(__file__).parent, 'configs') 