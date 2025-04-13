import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def draw_boxes(image, boxes, labels, color=(0, 255, 0), thickness=2):
    """在图像上绘制边界框"""
    img_copy = image.copy()
    
    for box in boxes:
        # 边界框坐标 - YOLO格式: [class_id, cx, cy, w, h]
        if len(box) == 5:
            class_id, cx, cy, w, h = box
            class_id = int(class_id)  # 确保class_id为整数
            
            # 转换为像素坐标
            x1 = int((cx - w/2) * image.shape[1])
            y1 = int((cy - h/2) * image.shape[0])
            x2 = int((cx + w/2) * image.shape[1])
            y2 = int((cy + h/2) * image.shape[0])
            
            # 防止坐标超出图像边界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1]-1, x2)
            y2 = min(image.shape[0]-1, y2)
            
            # 根据类别选择不同颜色
            box_color = color
            if class_id == 0:
                box_color = (0, 255, 0)  # 绿色
            elif class_id == 1:
                box_color = (255, 0, 0)  # 红色
            elif class_id == 2:
                box_color = (0, 0, 255)  # 蓝色
            elif class_id == 3:
                box_color = (255, 255, 0)  # 黄色
            
            # 绘制矩形
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, thickness)
            
            # 绘制标签
            cv2.putText(
                img_copy, 
                f"细胞{class_id}", 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                box_color, 
                thickness
            )
    
    return img_copy

def load_annotation(ann_path):
    """加载YOLO格式的标注"""
    boxes = []
    
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    box = [float(p) for p in parts]
                    boxes.append(box)
    
    return boxes

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试酵母细胞数据集")
    parser.add_argument('--config', type=str, default='configs/train/cell_data.yaml', help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=5, help='显示的样本数量')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 数据路径
    train_dir = Path(config['data']['train_dir'])
    ann_dir = Path(config['data']['ann_dir'])
    
    # 获取图像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(train_dir.glob(f'*{ext}')))
    
    if not image_files:
        print(f"未找到图像文件: {train_dir}")
        return
    
    print(f"找到 {len(image_files)} 张训练图像")
    
    # 显示样本
    sample_count = min(args.num_samples, len(image_files))
    plt.figure(figsize=(15, 5 * sample_count))
    
    for i in range(sample_count):
        img_path = image_files[i]
        ann_path = ann_dir / f"{img_path.stem}.txt"
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取标注
        boxes = load_annotation(ann_path)
        
        # 绘制边界框
        img_with_boxes = draw_boxes(img, boxes, [0] * len(boxes))
        
        # 显示图像
        plt.subplot(sample_count, 1, i + 1)
        plt.imshow(img_with_boxes)
        plt.title(f"图像: {img_path.name}, 检测到 {len(boxes)} 个细胞")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    print(f"已保存样本图像到 dataset_samples.png")
    
    # 打印统计信息
    total_boxes = 0
    for img_path in image_files:
        ann_path = ann_dir / f"{img_path.stem}.txt"
        boxes = load_annotation(ann_path)
        total_boxes += len(boxes)
    
    print(f"训练集中共有 {total_boxes} 个标注的细胞")

if __name__ == '__main__':
    main() 