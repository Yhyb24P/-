#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练脚本

此脚本用于调用酵母细胞检测训练模块，为统一接口设计。
"""

import os
import sys
import argparse

# 确保路径存在于Python路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='酵母细胞检测训练脚本')
    parser.add_argument('--config', type=str, default='configs/train/cell_data.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--visualize', action='store_true', help='启用增强可视化功能')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小')
    parser.add_argument('--device', type=str, default=None, help='训练设备，如cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--high-res', action='store_true', help='启用高分辨率模式')
    parser.add_argument('--efficient-train', action='store_true', help='启用高效训练模式')
    parser.add_argument('--multi-scale', action='store_true', help='启用多尺度训练')
    parser.add_argument('--high-res-size', type=int, default=1280, help='高分辨率模式的图像尺寸')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 构建命令行参数
    cmd_args = [
        f"--config {args.config}",
    ]
    
    if args.resume:
        cmd_args.append(f"--resume {args.resume}")
    
    if args.visualize:
        cmd_args.append("--visualize")
        
    if args.amp:
        cmd_args.append("--amp")
        
    if args.batch_size:
        cmd_args.append(f"--batch-size {args.batch_size}")
        
    if args.device:
        cmd_args.append(f"--device {args.device}")
        
    if args.high_res:
        cmd_args.append("--high-res")
        
    if args.efficient_train:
        cmd_args.append("--efficient-train")
        
    if args.multi_scale:
        cmd_args.append("--multi-scale")
        
    if args.high_res_size != 1280:
        cmd_args.append(f"--high-res-size {args.high_res_size}")
    
    # 执行训练脚本
    cmd = f"python {os.path.join(project_dir, 'train_cell.py')} {' '.join(cmd_args)}"
    print(f"执行命令: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
