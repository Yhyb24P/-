#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像增强主接口

提供显微镜下酵母细胞图像增强的命令行接口
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# 导入增强模块
from celldetection.enhance.enhance import enhance_image, enhance_images_batch

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='酵母细胞显微图像增强工具')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='输入图像路径 (单个图像或目录)')
    parser.add_argument('--output', '-o', type=str, default='./enhanced_output',
                      help='输出目录')
    parser.add_argument('--method', '-m', type=str, default='adaptive',
                      choices=['adaptive', 'clahe', 'adaptive_clahe', 'guided', 'basic'],
                      help='增强方法')
    parser.add_argument('--small-cell', '-s', action='store_true',
                      help='启用小细胞专用增强')
    parser.add_argument('--denoise', '-d', type=int, default=5,
                      help='降噪强度 (0-10), 0表示不降噪')
    parser.add_argument('--comparison', '-c', action='store_true',
                      help='创建原图与增强效果对比图')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                      help='CLAHE算法对比度限制参数 (仅用于clahe方法)')
    parser.add_argument('--tile-size', type=int, default=8,
                      help='CLAHE算法网格大小 (仅用于clahe方法)')
    parser.add_argument('--radius', type=int, default=2,
                      help='引导滤波半径 (仅用于guided方法)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                      help='引导滤波正则化参数 (仅用于guided方法)')
    parser.add_argument('--alpha', type=float, default=1.2,
                      help='对比度调整参数 (仅用于basic方法)')
    parser.add_argument('--beta', type=int, default=10,
                      help='亮度调整参数 (仅用于basic方法)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 收集方法特定参数
    params = {}
    
    if args.method == 'clahe':
        params['clip_limit'] = args.clip_limit
        params['tile_grid_size'] = (args.tile_size, args.tile_size)
    elif args.method == 'guided':
        params['radius'] = args.radius
        params['eps'] = args.epsilon
    elif args.method == 'basic':
        params['alpha'] = args.alpha
        params['beta'] = args.beta
    
    # 批量处理图像
    try:
        output_paths = enhance_images_batch(
            input_paths=args.input,
            output_dir=args.output,
            method=args.method,
            small_cell_enhancement=args.small_cell,
            denoise_level=args.denoise,
            create_comparison=args.comparison,
            params=params
        )
        
        print(f"增强处理完成. 共处理 {len(output_paths)} 张图像.")
        print(f"结果已保存到: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 