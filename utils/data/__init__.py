"""
数据处理工具模块

此模块从core.data.augment导入数据增强功能
"""

# 直接从core.data.augment导入
try:
    from core.data.augment import (
        YeastAugmentation,
        MosaicAugmentation, 
        get_transforms,
        get_augmentation
    )
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"无法导入数据增强功能，请确认core.data.augment模块存在: {e}")
    
    # 提供占位类和函数以防导入失败
    class YeastAugmentation:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return {"image": None, "bboxes": None}
    
    class MosaicAugmentation:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return None, None, None
    
    def get_transforms(*args, **kwargs):
        return None
        
    def get_augmentation(*args, **kwargs):
        return None
        
__all__ = [
    'YeastAugmentation',
    'MosaicAugmentation',
    'get_transforms',
    'get_augmentation'
] 