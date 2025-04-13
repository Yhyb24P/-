#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测项目的数据增强模块
用于替代旧版的utils.data.augmentation模块
"""

import numpy as np
import cv2

class BaseAugmentation:
    """增强基类"""
    def __init__(self):
        pass
    
    def __call__(self, image, **kwargs):
        """应用增强"""
        return self.apply(image, **kwargs)
    
    def apply(self, image, **kwargs):
        """子类实现具体增强方法"""
        raise NotImplementedError

class RandomBrightness(BaseAugmentation):
    """随机亮度增强"""
    def __init__(self, brightness_factor=0.3):
        super().__init__()
        self.brightness_factor = brightness_factor
        
    def apply(self, image, **kwargs):
        """应用亮度增强"""
        factor = np.random.uniform(1-self.brightness_factor, 1+self.brightness_factor)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return {"image": image, **kwargs}

class RandomContrast(BaseAugmentation):
    """随机对比度增强"""
    def __init__(self, contrast_factor=0.3):
        super().__init__()
        self.contrast_factor = contrast_factor
        
    def apply(self, image, **kwargs):
        """应用对比度增强"""
        factor = np.random.uniform(1-self.contrast_factor, 1+self.contrast_factor)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return {"image": image, **kwargs}

class Compose:
    """组合多个增强操作"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, **kwargs):
        """依次应用所有增强"""
        data = {"image": image, **kwargs}
        for t in self.transforms:
            data = t(**data)
        return data 