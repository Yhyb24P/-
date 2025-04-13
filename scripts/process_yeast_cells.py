#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立版本的预处理脚本，避免包导入问题
将所有必要的功能嵌入到一个文件中
"""

import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import shutil
import random
import albumentations as A

# 实用函数：图像加载、预处理、增强对比度和检测细胞

def load_image(image_path):
    """加载图像"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image, target_size=(512, 512), normalize=True):
    """预处理图像"""
    # 调整大小
    image = cv2.resize(image, target_size)
    
    # 归一化
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image

def enhance_contrast(image):
    """增强图像对比度"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 直方图均衡化
    enhanced = cv2.equalizeHist(gray)
    
    # 转回RGB
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced

def detect_cells(image, min_area=100, max_area=1000):
    """检测酵母细胞"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 过滤轮廓
    cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cells.append((x, y, w, h))
    
    return cells

def visualize_detection(image, cells, save_path=None, color=(0, 255, 0), thickness=2, show_count=True):
    """
    使用OpenCV可视化细胞检测结果
    """
    # 复制图像以免修改原图
    vis_image = image.copy()
    
    # 绘制检测框
    for i, (x, y, w, h) in enumerate(cells):
        # 绘制矩形
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # 显示ID
        cv2.putText(vis_image, f"#{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1, cv2.LINE_AA)
    
    # 显示总数
    if show_count:
        count_text = f"细胞数量: {len(cells)}"
        cv2.putText(vis_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    
    # 保存结果
    if save_path:
        # 确保目录存在
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转为BGR并保存
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

# 数据增强类

class YeastAugmentation:
    def __init__(self, config_path=None):
        """酵母细胞图像增强类
        
        Args:
            config_path: 配置文件路径，若提供则从配置文件加载参数
        """
        # 默认参数
        self.augmentation_intensity = "medium"  # 可选: "weak", "medium", "strong"
        self.mixup_prob = 0.15  # MixUp概率
        self.cached_images = []  # 用于MixUp的图像缓存
        self.max_cache_size = 30  # 最大缓存图像数
        
        # 标准化参数
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.fill_value = (114, 114, 114)  # 灰色填充
        
        # 从配置文件加载参数（如果提供）
        if config_path:
            self._load_config(config_path)
        
        # 根据强度级别调整参数
        self._adjust_params_by_intensity()
            
        # 创建训练增强管道
        self.train_transform = self._create_train_transforms()
        
        # 创建验证/测试增强管道（仅预处理）
        self.val_transform = self._create_val_transforms()
    
    def _create_train_transforms(self):
        """创建训练数据增强转换管道"""
        return A.Compose([
            # 1. 预处理与标准化步骤
            A.LongestMaxSize(max_size=512),  # 保持纵横比的尺寸限制
            A.PadIfNeeded(
                min_height=512, 
                min_width=512, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=self.fill_value  # 使用灰色填充
            ),
            
            # 2. 色彩空间增强 - 根据论文使用概率触发
            A.ColorJitter(
                brightness=0.1,      # 亮度调整幅度 ±10%
                contrast=0.1,        # 对比度调整幅度 ±10%
                saturation=0.1,      # 饱和度调整幅度 ±10%
                hue=0.02,            # 色调偏移幅度 ±0.02
                p=0.5                # 50%概率触发
            ),
            A.HueSaturationValue(
                hue_shift_limit=2,   # 对应论文的hsv_h=0.02
                sat_shift_limit=70,  # 对应论文的hsv_s=0.7
                val_shift_limit=0,   
                p=0.3
            ),
            
            # 3. 空间变换增强
            A.ShiftScaleRotate(
                shift_limit=0.05,    # 最大平移5%
                scale_limit=0.1,     # 缩放范围±10%
                rotate_limit=10,     # 旋转角度±10°
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3                # 30%概率触发
            ),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.VerticalFlip(p=0.5),    # 垂直翻转
            
            # 4. 高级增强技术 - 局部扰动
            A.Blur(blur_limit=3, p=0.1),          # 高斯模糊
            A.MotionBlur(blur_limit=7, p=0.2),    # 动态运动模糊
            
            # 最后应用标准化和Tensor转换
            A.Normalize(
                mean=self.mean,
                std=self.std
            ),
            A.ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels']
        ) if self.use_bbox else None)
    
    def _create_val_transforms(self):
        """创建验证/测试数据转换管道 - 仅基础预处理"""
        return A.Compose([
            A.LongestMaxSize(max_size=512),   
            A.PadIfNeeded(
                min_height=512, 
                min_width=512, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=self.fill_value
            ),
            A.Normalize(
                mean=self.mean,
                std=self.std
            ),
            A.ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels']
        ) if self.use_bbox else None)
        
    def __call__(self, image, bboxes=None, class_labels=None, is_train=True):
        """应用增强
        
        Args:
            image: 输入图像 (H,W,C)
            bboxes: YOLO格式边界框 [x_center, y_center, width, height]
            class_labels: 类别标签
            is_train: 是否用于训练（如为False则使用val_transform）
            
        Returns:
            增强后的图像和边界框
        """
        # 设置参数
        params = {
            'image': image
        }
        if bboxes is not None and class_labels is not None and self.use_bbox:
            params['bboxes'] = bboxes
            params['class_labels'] = class_labels
            
        # 应用适当的转换
        if is_train:
            transform = self.train_transform
        else:
            transform = self.val_transform
            
        # 应用albumentations增强
        augmented = transform(**params)
        
        # 应用MixUp (仅对训练图像，有一定概率)
        if is_train and random.random() < self.mixup_prob and len(self.cached_images) > 0:
            # 从缓存中随机选择一张图像进行混合
            cache_idx = random.randint(0, len(self.cached_images) - 1)
            mix_img = self.cached_images[cache_idx]
            
            # 执行MixUp - 随机混合比例
            alpha = random.uniform(0.3, 0.7)
            augmented['image'] = alpha * augmented['image'] + (1 - alpha) * mix_img
        
        # 更新图像缓存
        if is_train and len(self.cached_images) < self.max_cache_size:
            self.cached_images.append(augmented['image'].clone() if hasattr(augmented['image'], 'clone') else augmented['image'].copy())
        elif is_train and len(self.cached_images) >= self.max_cache_size and random.random() < 0.2:
            # 随机替换缓存中的图像
            replace_idx = random.randint(0, len(self.cached_images) - 1)
            self.cached_images[replace_idx] = augmented['image'].clone() if hasattr(augmented['image'], 'clone') else augmented['image'].copy()
        
        return augmented
    
    def _adjust_params_by_intensity(self):
        """根据增强强度调整参数"""
        # 初始化边界框使用标志
        self.use_bbox = False
        
        if self.augmentation_intensity == "weak":
            # 弱增强
            self.mixup_prob = 0.05
            self.max_cache_size = 20
        elif self.augmentation_intensity == "medium":
            # 中等增强
            self.mixup_prob = 0.15
            self.max_cache_size = 30
        elif self.augmentation_intensity == "strong":
            # 强力增强
            self.mixup_prob = 0.25
            self.max_cache_size = 50
    
    def _load_config(self, config_path):
        """从配置文件加载参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 如果配置中包含数据增强部分，则加载参数
            if 'data_augmentation' in config:
                aug_config = config['data_augmentation']
                
                # 加载各种参数
                if 'intensity' in aug_config:
                    self.augmentation_intensity = aug_config['intensity']
                if 'mixup_prob' in aug_config:
                    self.mixup_prob = aug_config['mixup_prob']
                if 'hsv_h' in aug_config:
                    self.hsv_h = aug_config['hsv_h']
                if 'hsv_s' in aug_config:
                    self.hsv_s = aug_config['hsv_s']
                if 'mean' in aug_config:
                    self.mean = aug_config['mean']
                if 'std' in aug_config:
                    self.std = aug_config['std']
                if 'fill_value' in aug_config:
                    self.fill_value = aug_config['fill_value']
                if 'use_bbox' in aug_config:
                    self.use_bbox = aug_config['use_bbox']
                    
            print(f"已从 {config_path} 加载数据增强配置")
        except Exception as e:
            print(f"加载配置文件时发生错误: {e}")
            print("使用默认配置继续...")
            
    def visualize_augmentations(self, image, num_examples=9):
        """生成图像增强示例网格"""
        # 首先预处理原始图像到标准尺寸
        image_resized = cv2.resize(image.copy(), (512, 512))
        
        # 创建一个3x3网格展示增强效果
        rows = cols = int(np.ceil(np.sqrt(num_examples)))
        height, width = 512, 512  # 使用固定尺寸
        grid = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
        
        # 第一张放原始图像（已调整大小）
        grid[:height, :width, :] = image_resized
        
        # 生成增强示例
        for i in range(1, num_examples):
            row = i // cols
            col = i % cols
            
            # 应用增强
            aug_result = self(image.copy(), is_train=True)
            aug_image = aug_result['image']
            
            # 如果输出是tensor，转回numpy数组
            if hasattr(aug_image, 'permute'):
                # PyTorch Tensor (C,H,W) -> numpy array (H,W,C)
                aug_image = aug_image.permute(1, 2, 0).numpy()
                
                # 反标准化
                aug_image = aug_image * np.array(self.std) + np.array(self.mean)
                aug_image = np.clip(aug_image * 255, 0, 255).astype(np.uint8)
            
            # 确保所有图像大小一致
            aug_image_resized = cv2.resize(aug_image, (512, 512))
            
            # 将增强后的图像添加到网格
            grid[row*height:(row+1)*height, col*width:(col+1)*width, :] = aug_image_resized
        
        return grid

# 辅助函数

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='酵母细胞图像预处理脚本')
    
    # 输入输出
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='原始图像目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    
    # 增强相关
    parser.add_argument('--aug_intensity', type=str, default='medium',
                        choices=['weak', 'medium', 'strong'],
                        help='数据增强强度')
    parser.add_argument('--aug_examples', type=int, default=0,
                        help='生成的增强示例数量')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='禁用数据增强')
    
    # 数据集分割
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    
    # 其他选项
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化结果')
    parser.add_argument('--config', type=str, default='configs/augmentation.yaml',
                        help='配置文件路径')
    
    return parser.parse_args()

def load_config(config_path=None):
    """加载配置
    
    Args:
        config_path: 配置文件路径，默认为None
        
    Returns:
        配置字典
    """
    # 默认配置
    config = {
        'raw_dir': 'data/raw',
        'output_dir': 'data/processed',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'image': {
            'target_size': (512, 512),
            'enhance_contrast': True
        },
        'detection': {
            'min_area': 100,
            'max_area': 1000
        },
        'data_augmentation': {
            'intensity': 'medium',
            'mixup_prob': 0.15,
            'hsv_h': 0.02,
            'hsv_s': 0.7,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'fill_value': (114, 114, 114)
        }
    }
    
    # 从文件加载配置
    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                
            # 更新配置
            for key, value in file_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    # 递归更新嵌套字典
                    config[key].update(value)
                else:
                    config[key] = value
                    
            print(f"从 {config_path} 加载配置成功")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
    
    return config

def create_directories(config):
    """创建所需的目录结构
    
    Args:
        config: 配置字典
    """
    output_dir = Path(config.get('output_dir', 'data/processed'))
    
    # 创建基本目录
    for subset in ['train', 'val', 'test']:
        (output_dir / subset).mkdir(parents=True, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = Path(config.get('visualization', {}).get('output_dir', 'data/visualization'))
    for subset in ['train', 'val', 'test']:
        (vis_dir / subset).mkdir(parents=True, exist_ok=True)
    
    # 创建增强示例目录
    (output_dir / 'augmentation_examples').mkdir(parents=True, exist_ok=True)
    
    # 创建配置目录
    Path('configs').mkdir(exist_ok=True)
    
    # 如果配置文件不存在，创建默认配置文件
    default_config_path = Path('configs/augmentation.yaml')
    if not default_config_path.exists():
        default_config = {
            'data_augmentation': {
                'intensity': 'medium',
                'mixup_prob': 0.15,
                'hsv_h': 0.02,
                'hsv_s': 0.7,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'fill_value': [114, 114, 114]
            }
        }
        with open(default_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"创建默认配置文件: {default_config_path}")
        
    print(f"创建目录结构: {output_dir}")

def process_raw_images(config, visualize=False, aug_examples=0, aug_intensity='medium', no_augmentation=False):
    """处理原始图像

    Args:
        config: 配置字典
        visualize: 是否生成可视化结果
        aug_examples: 为每个图像生成的增强示例数量
        aug_intensity: 增强强度 ('weak', 'medium', 'strong')
        no_augmentation: 是否禁用增强
    """
    print("\n===== 开始处理原始图像 =====")
    
    # 获取原始图像目录
    raw_dir = Path(config.get('raw_dir', 'data/raw'))
    if not raw_dir.exists():
        print(f"错误: 原始图像目录 {raw_dir} 不存在")
        print("正在创建目录...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"请将原始图像放入 {raw_dir} 目录，然后重新运行脚本")
        return

    # 获取输出目录
    output_dir = Path(config.get('output_dir', 'data/processed'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 找到所有图像
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    print(f"搜索支持的图像格式: {', '.join(extensions)}")
    
    image_paths = []
    for ext in extensions:
        found = list(raw_dir.glob(f'**/*{ext}'))
        print(f"找到 {len(found)} 张 {ext} 格式图像")
        image_paths.extend(found)
    
    print(f"总共找到 {len(image_paths)} 张图像")
    if len(image_paths) == 0:
        print(f"警告: 在 {raw_dir} 中没有找到任何图像文件")
        return

    # 创建输出目录
    print("\n===== 创建目录结构 =====")
    create_directories(config)

    # 创建数据增强器
    print("\n===== 配置数据增强 =====")
    augmenter = None
    if not no_augmentation:
        try:
            config_path = Path(config.get('config_path', 'configs/augmentation.yaml'))
            print(f"使用配置文件: {config_path if config_path.exists() else '默认配置'}")
            
            augmenter = YeastAugmentation(config_path if config_path.exists() else None)
            augmenter.augmentation_intensity = aug_intensity
            augmenter._adjust_params_by_intensity()
            
            print(f"数据增强已启用 - 强度: {aug_intensity}")
            print(f"MixUp概率: {augmenter.mixup_prob:.2f}")
            if aug_examples > 0:
                print(f"将为每张图像生成 {aug_examples} 个增强示例")
        except Exception as e:
            print(f"警告: 创建增强器时出错: {e}")
            print("将继续处理但不使用数据增强")
            augmenter = None
            no_augmentation = True
    else:
        print("数据增强已禁用")
    
    # 根据比例分割训练/验证/测试集
    print("\n===== 分割数据集 =====")
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    
    # 确保比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"警告: 数据集比例总和({total_ratio})不等于1，将进行归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    print(f"训练集比例: {train_ratio:.2f}")
    print(f"验证集比例: {val_ratio:.2f}")
    print(f"测试集比例: {test_ratio:.2f}")
    
    # 打乱图像顺序
    print("随机打乱数据集...")
    random.shuffle(image_paths)
    
    # 计算每个集合的大小
    total_images = len(image_paths)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    
    # 分割数据集
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:]
    
    print(f"数据集分割完成: 训练集({len(train_paths)}张), 验证集({len(val_paths)}张), 测试集({len(test_paths)}张)")
    
    # 处理每个子集
    print("\n===== 开始处理各子集 =====")
    
    try:
        print("\n----- 处理训练集 -----")
        process_subset('train', train_paths, config, visualize, augmenter, aug_examples)
        
        print("\n----- 处理验证集 -----")
        process_subset('val', val_paths, config, visualize, augmenter, 0)  # 验证集不需要生成增强示例
        
        print("\n----- 处理测试集 -----")
        process_subset('test', test_paths, config, visualize, augmenter, 0)  # 测试集不需要生成增强示例
        
        print("\n===== 预处理完成! =====")
        print(f"处理结果已保存到 {output_dir}")
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("尽管发生错误，已完成的处理结果仍然有效")

def process_subset(subset_name, image_paths, config, visualize, augmenter=None, aug_examples=0):
    """处理数据子集
    
    Args:
        subset_name: 子集名称 ('train', 'val', 'test')
        image_paths: 图像路径列表
        config: 配置字典
        visualize: 是否生成可视化结果
        augmenter: 数据增强器
        aug_examples: 为每个图像生成的增强示例数量
    """
    # 获取输出目录
    output_dir = Path(config.get('output_dir', 'data/processed'))
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = Path(config.get('visualization', {}).get('output_dir', 'data/visualization'))
    for subset in ['train', 'val', 'test']:
        (vis_dir / subset).mkdir(parents=True, exist_ok=True)
    
    # 创建增强示例目录
    aug_dir = None
    if aug_examples > 0 and subset_name == 'train':
        aug_dir = output_dir / 'augmentation_examples'
        aug_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图像
    print(f"处理 {subset_name} 集合中的 {len(image_paths)} 张图像...")
    
    for img_path in tqdm(image_paths):
        # 加载图像
        try:
            image = load_image(img_path)
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            continue
        
        # 检测细胞
        cells = detect_cells(image)
        
        # 生成文件名
        filename = img_path.stem
        processed_path = subset_dir / f"{filename}.png"
        
        # 确定是否是训练集
        is_train = subset_name == 'train'
        
        # 应用数据增强
        processed_image = image.copy()
        if augmenter is not None:
            try:
                # 创建边界框和标签格式
                bboxes = []
                class_labels = []
                for x, y, w, h in cells:
                    # 转换为YOLO格式 [x_center, y_center, width, height]
                    img_h, img_w = image.shape[:2]
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    bboxes.append([x_center, y_center, norm_w, norm_h])
                    class_labels.append(0)  # 假设只有一个类别
                
                # 应用增强 - 对于训练集使用完整增强，对于验证/测试集仅使用预处理
                augmented = augmenter(image, bboxes, class_labels, is_train=is_train)
                processed_image = augmented['image']
                
                # 处理增强后图像 - 如果是tensor则转换回numpy
                if hasattr(processed_image, 'permute'):
                    # PyTorch Tensor (C,H,W) -> numpy array (H,W,C)
                    processed_image = processed_image.permute(1, 2, 0).numpy()
                    
                    # 反标准化
                    processed_image = processed_image * np.array(augmenter.std) + np.array(augmenter.mean)
                    processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"应用增强失败: {e}，使用预处理图像")
                processed_image = preprocess_image(image)
        else:
            # 如果没有增强器，简单地预处理图像
            processed_image = preprocess_image(image)
        
        # 保存处理后的图像
        cv2.imwrite(str(processed_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        
        # 生成可视化结果
        if visualize and vis_dir:
            # 在原始图像上标记细胞位置
            vis_image = visualize_detection(image, cells)
            vis_path = vis_dir / subset_name / f"{filename}_detection.png"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            # 保存处理后的图像供比较
            proc_vis_path = vis_dir / subset_name / f"{filename}_processed.png"
            cv2.imwrite(str(proc_vis_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        
        # 生成增强示例
        if aug_examples > 0 and augmenter is not None and aug_dir and is_train:
            try:
                # 生成增强示例网格
                aug_grid = augmenter.visualize_augmentations(image, num_examples=aug_examples)
                aug_path = aug_dir / f"{filename}_augmentations.png"
                cv2.imwrite(str(aug_path), cv2.cvtColor(aug_grid, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"生成增强示例失败: {e}")
    
    print(f"{subset_name} 集合处理完成，已保存到 {subset_dir}")

def main():
    """主函数：解析参数并执行预处理"""
    try:
        print("\n============================================")
        print("       酵母细胞图像预处理与数据增强工具       ")
        print("============================================\n")
        
        # 检查版本
        print(f"Python版本: {sys.version}")
        print(f"OpenCV版本: {cv2.__version__}")
        print(f"NumPy版本: {np.__version__}")
        try:
            import torch
            print(f"PyTorch版本: {torch.__version__}")
        except ImportError:
            print("PyTorch未安装 (非必需)")
            
        try:
            import albumentations
            print(f"Albumentations版本: {albumentations.__version__}")
        except (ImportError, AttributeError):
            print("警告: Albumentations库未安装或版本过低")
            print("请运行: pip install albumentations>=1.0.0")
            return
        
        # 解析命令行参数
        args = parse_args()
        print("\n===== 配置信息 =====")
        print(f"配置文件: {args.config}")
        print(f"原始图像目录: {args.raw_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"增强强度: {args.aug_intensity}")
        print(f"增强示例数量: {args.aug_examples}")
        print(f"是否生成可视化: {'是' if args.visualize else '否'}")
        print(f"是否禁用增强: {'是' if args.no_augmentation else '否'}")
        
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖配置文件
        if args.raw_dir:
            config['raw_dir'] = args.raw_dir
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.train_ratio:
            config['train_ratio'] = args.train_ratio
        if args.val_ratio:
            config['val_ratio'] = args.val_ratio
        if args.test_ratio:
            config['test_ratio'] = args.test_ratio
        
        # 创建目录
        create_directories(config)
        
        # 处理原始图像
        process_raw_images(
            config, 
            args.visualize, 
            args.aug_examples, 
            args.aug_intensity,
            args.no_augmentation
        )
        
        print("\n预处理完成，感谢使用！\n")
        
    except KeyboardInterrupt:
        print("\n用户中断处理，已停止")
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n如需帮助，请查看错误信息")
    
if __name__ == '__main__':
    main() 