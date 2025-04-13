import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

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

def analyze_dataset(config):
    """分析数据集统计信息"""
    # 数据路径
    train_dir = Path(config['data']['train_dir'])
    val_dir = Path(config['data']['val_dir'])
    test_dir = Path(config['data']['test_dir'])
    ann_dir = Path(config['data']['ann_dir'])
    
    # 创建统计字典
    stats = {
        'train': {'images': 0, 'annotations': 0, 'sizes': []},
        'val': {'images': 0, 'annotations': 0, 'sizes': []},
        'test': {'images': 0, 'annotations': 0, 'sizes': []},
        'box_sizes': [],
        'aspect_ratios': []
    }
    
    # 分析各个数据集
    for split, dir_path in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        if not dir_path.exists():
            continue
            
        # 获取图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(dir_path.glob(f'*{ext}')))
        
        stats[split]['images'] = len(image_files)
        
        # 处理每个图像
        for img_path in image_files:
            ann_path = ann_dir / f"{img_path.stem}.txt"
            boxes = load_annotation(ann_path)
            stats[split]['annotations'] += len(boxes)
            
            # 计算边界框尺寸和宽高比
            for box in boxes:
                _, _, _, w, h = box
                stats['box_sizes'].append(w * h)  # 面积
                if h > 0:
                    stats['aspect_ratios'].append(w / h)  # 宽高比
            
            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                stats[split]['sizes'].append((w, h))
    
    return stats

def plot_statistics(stats, output_dir='.'):
    """绘制数据集统计图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据集分布
    splits = ['train', 'val', 'test']
    image_counts = [stats[s]['images'] for s in splits]
    annotation_counts = [stats[s]['annotations'] for s in splits]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图像数量
    ax1.bar(splits, image_counts, color=['blue', 'green', 'red'])
    ax1.set_title('图像数量分布')
    ax1.set_ylabel('图像数量')
    
    # 标注数量
    ax2.bar(splits, annotation_counts, color=['blue', 'green', 'red'])
    ax2.set_title('标注数量分布')
    ax2.set_ylabel('标注数量')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'))
    
    # 2. 边界框尺寸分布
    plt.figure(figsize=(8, 6))
    plt.hist(stats['box_sizes'], bins=20)
    plt.title('边界框面积分布')
    plt.xlabel('标准化面积')
    plt.ylabel('频率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'box_sizes.png'))
    
    # 3. 宽高比分布
    plt.figure(figsize=(8, 6))
    plt.hist(stats['aspect_ratios'], bins=20)
    plt.title('边界框宽高比分布')
    plt.xlabel('宽高比')
    plt.ylabel('频率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'aspect_ratios.png'))
    
    # 4. 保存统计数据
    summary = {
        'train_images': stats['train']['images'],
        'val_images': stats['val']['images'],
        'test_images': stats['test']['images'],
        'train_annotations': stats['train']['annotations'],
        'val_annotations': stats['val']['annotations'],
        'test_annotations': stats['test']['annotations'],
        'total_images': sum(image_counts),
        'total_annotations': sum(annotation_counts),
        'avg_annotations_per_image': sum(annotation_counts) / sum(image_counts) if sum(image_counts) > 0 else 0,
        'avg_box_size': np.mean(stats['box_sizes']) if stats['box_sizes'] else 0,
        'avg_aspect_ratio': np.mean(stats['aspect_ratios']) if stats['aspect_ratios'] else 0
    }
    
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="数据集统计分析")
    parser.add_argument('--config', type=str, default='configs/train/cell_data.yaml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='data/stats', help='输出目录')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 分析数据集
    stats = analyze_dataset(config)
    
    # 绘制统计图表
    summary = plot_statistics(stats, args.output_dir)
    
    # 打印摘要
    print("\n===== 数据集统计摘要 =====")
    print(f"训练集: {summary['train_images']} 张图像, {summary['train_annotations']} 个标注")
    print(f"验证集: {summary['val_images']} 张图像, {summary['val_annotations']} 个标注")
    print(f"测试集: {summary['test_images']} 张图像, {summary['test_annotations']} 个标注")
    print(f"总计: {summary['total_images']} 张图像, {summary['total_annotations']} 个标注")
    print(f"平均每张图像的标注数量: {summary['avg_annotations_per_image']:.2f}")
    print(f"平均边界框面积: {summary['avg_box_size']:.4f}")
    print(f"平均边界框宽高比: {summary['avg_aspect_ratio']:.2f}")
    print("==========================")
    
    print(f"统计数据和图表已保存到 {args.output_dir} 目录")

if __name__ == '__main__':
    main()