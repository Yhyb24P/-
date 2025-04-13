#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞数据处理工具 - 简化版主入口
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='酵母细胞数据处理工具')
    parser.add_argument('--mode', type=str, default='preprocess', 
                        choices=['preprocess', 'visualize', 'augment'],
                        help='操作模式: preprocess-预处理, visualize-可视化, augment-数据增强')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='原始图像目录')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='输出根目录')
    parser.add_argument('--img_size', type=int, default=640,
                        help='目标图像尺寸')
    parser.add_argument('--aug_intensity', type=str, default='medium',
                        choices=['weak', 'medium', 'strong'],
                        help='数据增强强度')
    parser.add_argument('--aug_examples', type=int, default=0,
                        help='每张图像生成的增强示例数量')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化处理结果')
    
    args = parser.parse_args()
    
    # 导入处理脚本
    sys.path.append('scripts')
    try:
        from process_yeast_cells import (
            process_raw_images, load_config, create_directories, visualize_detection
        )
    except ImportError:
        print("错误: 无法导入process_yeast_cells模块，请确保scripts/process_yeast_cells.py文件存在")
        return 1
    
    # 创建默认配置
    config = {
        'paths': {
            'raw_data': args.raw_dir,
            'processed_data': os.path.join(args.output_dir, 'processed'),
            'annotations': os.path.join(args.output_dir, 'annotations'),
        },
        'image': {
            'target_size': [args.img_size, args.img_size],
            'enhance_contrast': True,
        },
        'detection': {
            'min_area': 100,
            'max_area': 2000,
        },
        'datasets': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
        },
        'augmentation': {
            'enabled': args.mode == 'augment',
            'intensity': args.aug_intensity,
        }
    }
    
    # 根据模式执行不同操作
    if args.mode == 'preprocess' or args.mode == 'augment':
        print(f"开始{'预处理' if args.mode == 'preprocess' else '增强'}原始图像...")
        create_directories(config)
        process_raw_images(
            config, 
            visualize=args.visualize, 
            aug_examples=args.aug_examples, 
            aug_intensity=args.aug_intensity,
            no_augmentation=(args.mode != 'augment')
        )
        print("处理完成!")
    elif args.mode == 'visualize':
        # 仅可视化模式
        print("开始可视化图像...")
        from scripts.process_yeast_cells import load_image, detect_cells
        
        # 获取所有原始图像
        raw_dir = Path(args.raw_dir)
        image_paths = list(raw_dir.glob('*.bmp')) + list(raw_dir.glob('*.BMP'))
        
        if not image_paths:
            print(f"在 {raw_dir} 中未找到BMP图像")
            return 1
        
        # 创建可视化输出目录
        vis_dir = Path(args.output_dir) / 'visualization'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理并可视化每张图像
        for image_path in image_paths:
            try:
                # 加载图像
                image = load_image(image_path)
                
                # 检测细胞
                cells = detect_cells(image, min_area=100, max_area=2000)
                
                # 可视化并保存
                output_path = vis_dir / f"{image_path.stem}_detected.jpg"
                visualize_detection(
                    image, 
                    cells, 
                    save_path=output_path, 
                    color=(0, 255, 0), 
                    thickness=2, 
                    show_count=True
                )
                print(f"已保存可视化结果: {output_path}")
                
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                continue
        
        print("可视化完成!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 