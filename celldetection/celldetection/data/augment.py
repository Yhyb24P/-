"""
数据增强模块

提供高级数据增强功能，包括MixUp、Mosaic和Cutout等。
特别针对细胞检测任务进行了优化。
"""

import cv2
import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional, Union, Any


class MixUp:
    """MixUp数据增强

    将两张图像按一定比例混合，同时混合它们的标签。
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        初始化MixUp增强器
        
        Args:
            alpha: Beta分布的参数，控制混合比例
        """
        self.alpha = alpha
        
    def __call__(self, 
                image1: np.ndarray, 
                boxes1: np.ndarray, 
                image2: np.ndarray, 
                boxes2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行MixUp增强
        
        Args:
            image1: 第一张图像
            boxes1: 第一张图像的边界框 [N, 5] (x1, y1, x2, y2, class_id)
            image2: 第二张图像
            boxes2: 第二张图像的边界框 [M, 5] (x1, y1, x2, y2, class_id)
            
        Returns:
            混合后的图像和边界框
        """
        # 确保图像尺寸一致
        h, w = image1.shape[:2]
        image2 = cv2.resize(image2, (w, h))
        
        # 生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合图像
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_image = mixed_image.astype(np.uint8)
        
        # 混合边界框 - 简单合并两组边界框，并附带混合权重
        if len(boxes1) > 0:
            boxes1 = np.concatenate([boxes1, np.full((len(boxes1), 1), lam)], axis=1)
        
        if len(boxes2) > 0:
            boxes2 = np.concatenate([boxes2, np.full((len(boxes2), 1), 1 - lam)], axis=1)
        
        # 合并边界框
        mixed_boxes = np.concatenate([boxes1, boxes2], axis=0) if len(boxes1) > 0 and len(boxes2) > 0 else \
                     boxes1 if len(boxes1) > 0 else boxes2
        
        return mixed_image, mixed_boxes


class Mosaic:
    """Mosaic数据增强

    将四张图像拼接成一张，增加小目标的数量和多样性。
    """
    
    def __init__(self, 
                output_size: Tuple[int, int] = (640, 640),
                border_value: int = 114):
        """
        初始化Mosaic增强器
        
        Args:
            output_size: 输出图像尺寸 (高度, 宽度)
            border_value: 填充边界的像素值
        """
        self.output_size = output_size
        self.border_value = border_value
        
    def __call__(self, 
                images: List[np.ndarray], 
                boxes_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行Mosaic增强
        
        Args:
            images: 四张输入图像的列表
            boxes_list: 四张图像对应的边界框列表
            
        Returns:
            拼接后的图像和调整后的边界框
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        
        # 输出图像尺寸
        output_h, output_w = self.output_size
        
        # 创建输出图像
        mosaic_img = np.full((output_h, output_w, 3), self.border_value, dtype=np.uint8)
        
        # 随机选择拼接点
        cx = int(random.uniform(output_w * 0.25, output_w * 0.75))
        cy = int(random.uniform(output_h * 0.25, output_h * 0.75))
        
        # 合并后的边界框列表
        merged_boxes = []
        
        # 处理四张图像
        for i, (img, boxes) in enumerate(zip(images, boxes_list)):
            h, w = img.shape[:2]
            
            # 根据位置确定放置区域
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = 0, 0, cx, cy
                x1b, y1b, x2b, y2b = w - cx, h - cy, w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = cx, 0, output_w, cy
                x1b, y1b, x2b, y2b = 0, h - cy, output_w - cx, h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = 0, cy, cx, output_h
                x1b, y1b, x2b, y2b = w - cx, 0, w, output_h - cy
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = cx, cy, output_w, output_h
                x1b, y1b, x2b, y2b = 0, 0, output_w - cx, output_h - cy
            
            # 放置图像
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # 调整边界框坐标
            if len(boxes) > 0:
                # 复制边界框
                adjusted_boxes = boxes.copy()
                
                # 调整坐标
                adjusted_boxes[:, [0, 2]] = adjusted_boxes[:, [0, 2]] - x1b + x1a
                adjusted_boxes[:, [1, 3]] = adjusted_boxes[:, [1, 3]] - y1b + y1a
                
                # 裁剪到图像边界内
                adjusted_boxes[:, 0] = np.clip(adjusted_boxes[:, 0], x1a, x2a)
                adjusted_boxes[:, 1] = np.clip(adjusted_boxes[:, 1], y1a, y2a)
                adjusted_boxes[:, 2] = np.clip(adjusted_boxes[:, 2], x1a, x2a)
                adjusted_boxes[:, 3] = np.clip(adjusted_boxes[:, 3], y1a, y2a)
                
                # 过滤无效框
                valid_indices = (adjusted_boxes[:, 2] > adjusted_boxes[:, 0]) & \
                               (adjusted_boxes[:, 3] > adjusted_boxes[:, 1])
                
                if np.any(valid_indices):
                    merged_boxes.append(adjusted_boxes[valid_indices])
        
        # 合并所有有效边界框
        if merged_boxes:
            merged_boxes = np.concatenate(merged_boxes, axis=0)
        else:
            merged_boxes = np.zeros((0, 5))
        
        return mosaic_img, merged_boxes


class Cutout:
    """Cutout数据增强

    随机遮挡图像的一部分，提高模型的鲁棒性。
    """
    
    def __init__(self, 
                n_holes: int = 1, 
                length: int = 40, 
                fill_value: int = 0):
        """
        初始化Cutout增强器
        
        Args:
            n_holes: 遮挡区域的数量
            length: 遮挡区域的边长
            fill_value: 填充值
        """
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行Cutout增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        h, w = image.shape[:2]
        result = image.copy()
        
        for _ in range(self.n_holes):
            # 随机选择遮挡中心点
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # 计算遮挡区域
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # 应用遮挡
            if len(image.shape) == 3:
                result[y1:y2, x1:x2, :] = self.fill_value
            else:
                result[y1:y2, x1:x2] = self.fill_value
                
        return result


class CellSpecificAugmentation:
    """细胞特异性数据增强

    针对细胞检测任务的特定增强方法。
    """
    
    def __init__(self, 
                intensity_range: Tuple[float, float] = (0.5, 1.5),
                contrast_range: Tuple[float, float] = (0.8, 1.2),
                blur_prob: float = 0.3,
                noise_prob: float = 0.3):
        """
        初始化细胞特异性增强器
        
        Args:
            intensity_range: 亮度调整范围
            contrast_range: 对比度调整范围
            blur_prob: 模糊概率
            noise_prob: 噪声概率
        """
        self.intensity_range = intensity_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行细胞特异性增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        result = image.copy().astype(np.float32)
        
        # 亮度调整
        intensity_factor = np.random.uniform(*self.intensity_range)
        result = result * intensity_factor
        
        # 对比度调整
        contrast_factor = np.random.uniform(*self.contrast_range)
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = (result - mean) * contrast_factor + mean
        
        # 随机模糊
        if np.random.random() < self.blur_prob:
            kernel_size = np.random.choice([3, 5])
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
            
        # 随机噪声
        if np.random.random() < self.noise_prob:
            noise_type = np.random.choice(['gaussian', 'salt_pepper'])
            
            if noise_type == 'gaussian':
                # 高斯噪声
                noise = np.random.normal(0, 10, result.shape).astype(np.float32)
                result = result + noise
            else:
                # 椒盐噪声
                salt_vs_pepper = 0.5
                amount = 0.01
                
                # 盐噪声 (白点)
                salt = np.random.random(result.shape[:2]) < (amount * salt_vs_pepper)
                if len(result.shape) == 3:
                    for i in range(result.shape[2]):
                        result[salt, i] = 255
                else:
                    result[salt] = 255
                    
                # 椒噪声 (黑点)
                pepper = np.random.random(result.shape[:2]) < (amount * (1 - salt_vs_pepper))
                if len(result.shape) == 3:
                    for i in range(result.shape[2]):
                        result[pepper, i] = 0
                else:
                    result[pepper] = 0
        
        # 裁剪到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


class CellDivisionAugmentation:
    """细胞分裂增强

    模拟细胞分裂过程，增加训练数据多样性。
    """
    
    def __init__(self, 
                division_prob: float = 0.3,
                min_overlap: float = 0.3,
                max_overlap: float = 0.7):
        """
        初始化细胞分裂增强器
        
        Args:
            division_prob: 应用分裂增强的概率
            min_overlap: 最小重叠比例
            max_overlap: 最大重叠比例
        """
        self.division_prob = division_prob
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        
    def __call__(self, 
                image: np.ndarray, 
                boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行细胞分裂增强
        
        Args:
            image: 输入图像
            boxes: 边界框 [N, 5] (x1, y1, x2, y2, class_id)
            
        Returns:
            增强后的图像和边界框
        """
        if len(boxes) == 0 or np.random.random() > self.division_prob:
            return image, boxes
            
        result_img = image.copy()
        result_boxes = boxes.copy()
        
        # 随机选择一个细胞进行分裂
        idx = np.random.randint(len(boxes))
        box = boxes[idx].astype(np.int32)
        x1, y1, x2, y2, class_id = box
        
        # 提取细胞ROI
        cell_roi = image[y1:y2, x1:x2].copy()
        
        if cell_roi.size == 0:
            return image, boxes
            
        # 计算细胞中心
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # 计算细胞宽高
        w, h = x2 - x1, y2 - y1
        
        # 随机选择分裂方向
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 计算分裂距离 (重叠程度)
        overlap_ratio = np.random.uniform(self.min_overlap, self.max_overlap)
        distance = int((1 - overlap_ratio) * max(w, h))
        
        # 计算新细胞位置
        new_cx = int(cx + distance * np.cos(angle))
        new_cy = int(cy + distance * np.sin(angle))
        
        # 确保新细胞在图像内
        h_img, w_img = image.shape[:2]
        new_x1 = max(0, new_cx - w // 2)
        new_y1 = max(0, new_cy - h // 2)
        new_x2 = min(w_img, new_cx + w // 2)
        new_y2 = min(h_img, new_cy + h // 2)
        
        # 如果新区域太小，则放弃
        if new_x2 - new_x1 < w // 2 or new_y2 - new_y1 < h // 2:
            return image, boxes
            
        # 调整细胞ROI大小以适应新位置
        new_cell_roi = cv2.resize(cell_roi, (new_x2 - new_x1, new_y2 - new_y1))
        
        # 随机变形新细胞 (模拟分裂过程)
        # 1. 随机旋转
        angle = np.random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((new_cell_roi.shape[1] // 2, new_cell_roi.shape[0] // 2), angle, 1)
        new_cell_roi = cv2.warpAffine(new_cell_roi, M, (new_cell_roi.shape[1], new_cell_roi.shape[0]))
        
        # 2. 随机形变
        pts1 = np.float32([[0, 0], [new_cell_roi.shape[1], 0], 
                           [0, new_cell_roi.shape[0]], [new_cell_roi.shape[1], new_cell_roi.shape[0]]])
        
        # 添加小的随机扰动
        pts2 = pts1 + np.random.uniform(-0.1, 0.1, pts1.shape) * np.array([new_cell_roi.shape[1], new_cell_roi.shape[0]])
        
        M = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
        new_cell_roi = cv2.warpPerspective(new_cell_roi, M, (new_cell_roi.shape[1], new_cell_roi.shape[0]))
        
        # 将新细胞放置到图像中
        # 使用alpha混合以实现更自然的效果
        alpha = 0.7
        roi = result_img[new_y1:new_y2, new_x1:new_x2].copy()
        result_img[new_y1:new_y2, new_x1:new_x2] = cv2.addWeighted(roi, 1-alpha, new_cell_roi, alpha, 0)
        
        # 添加新的边界框
        new_box = np.array([[new_x1, new_y1, new_x2, new_y2, class_id]])
        result_boxes = np.concatenate([result_boxes, new_box], axis=0)
        
        return result_img, result_boxes
