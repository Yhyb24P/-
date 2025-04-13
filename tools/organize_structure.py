#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件结构整理工具
根据README.md中定义的框架自动组织文件结构
"""

import os
import shutil
import argparse
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='整理酵母细胞检测项目文件结构')
    parser.add_argument('--dry-run', action='store_true', help='仅显示要执行的操作，不实际执行')
    parser.add_argument('--force', action='store_true', help='强制执行，不提示确认')
    return parser.parse_args()

def create_dir_structure(base_dir, structure, dry_run=False):
    """创建目录结构"""
    for dir_path in structure:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            if dry_run:
                print(f"将创建目录: {dir_path}")
            else:
                os.makedirs(full_path, exist_ok=True)
                print(f"已创建目录: {dir_path}")

def move_file(src, dst, dry_run=False):
    """移动文件"""
    # 确保目标目录存在
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
    
    # 检查源文件是否存在
    if not os.path.exists(src):
        print(f"警告: 源文件不存在: {src}")
        return False
    
    # 检查目标文件是否已存在
    if os.path.exists(dst):
        print(f"警告: 目标文件已存在: {dst}")
        return False
    
    # 移动文件
    if dry_run:
        print(f"将移动: {src} -> {dst}")
    else:
        try:
            shutil.move(src, dst)
            print(f"已移动: {src} -> {dst}")
        except Exception as e:
            print(f"移动文件出错 {src} -> {dst}: {str(e)}")
            return False
    
    return True

def create_empty_file(path, content="", dry_run=False):
    """创建空文件"""
    if os.path.exists(path):
        return
    
    if dry_run:
        print(f"将创建文件: {path}")
    else:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已创建文件: {path}")
        except Exception as e:
            print(f"创建文件出错 {path}: {str(e)}")

def main():
    """主函数"""
    args = parse_args()
    
    # 定义标准目录结构（严格按照README.md中的描述）
    standard_dirs = [
        "celldetection",
        "celldetection/models",
        "celldetection/data",
        "celldetection/enhance",
        "celldetection/utils",
        "scripts",
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "data/annotations",
        "data/datasets",
        "data/visualization",
        "docs"
    ]
    
    # 定义文件映射关系（现有文件 -> 标准位置）
    file_mappings = [
        # celldetection包
        ("celldetection/models/backbone/backbone.py", "celldetection/models/backbone.py"),
        ("celldetection/models/detection/head.py", "celldetection/models/heads.py"),
        ("celldetection/models/attention/attention.py", "celldetection/models/attention.py"),
        ("core/data/augment.py", "celldetection/data/transforms.py"),
        ("celldetection/enhance/adaptive.py", "celldetection/enhance/adaptive.py"),
        
        # 工具和配置
        ("tools/check_migration.py", "tools/check_migration.py"),  # 保持不变
        ("tools/convert_weights.py", "tools/convert_weights.py"),  # 保持不变
        ("tools/test_migration.py", "tools/test_migration.py"),    # 保持不变
        ("tools/organize_structure.py", "tools/organize_structure.py"),  # 保持不变
    ]
    
    # 需要创建的空文件（包含基本框架）
    empty_files = [
        ("celldetection/__init__.py", "# 酵母细胞检测包\n"),
        ("celldetection/models/__init__.py", "# 模型模块\n"),
        ("celldetection/models/neck.py", "# FPN网络实现\n"),
        ("celldetection/data/__init__.py", "# 数据处理模块\n"),
        ("celldetection/data/dataset.py", "# 数据集类\n"),
        ("celldetection/data/utils.py", "# 数据工具\n"),
        ("celldetection/enhance/__init__.py", "# 图像增强模块\n"),
        ("celldetection/enhance/guided_filter.py", "# 引导滤波\n"),
        ("celldetection/enhance/clahe.py", "# CLAHE增强\n"),
        ("celldetection/enhance/small_cell.py", "# 小细胞增强\n"),
        ("celldetection/utils/__init__.py", "# 通用工具模块\n"),
        ("celldetection/utils/metrics.py", "# 评估指标\n"),
        ("celldetection/utils/visualize.py", "# 可视化工具\n"),
        ("celldetection/utils/post_process.py", "# 后处理函数\n"),
        ("celldetection/train.py", "# 训练接口\n"),
        ("celldetection/detect.py", "# 检测接口\n"),
        ("celldetection/enhance.py", "# 增强接口\n"),
        ("scripts/train.py", "# 训练脚本\n"),
        ("scripts/detect.py", "# 检测脚本\n"),
        ("scripts/enhance.py", "# 单独增强脚本\n"),
        ("configs/default.yaml", "# 默认配置\n"),
        ("configs/train_configs.yaml", "# 训练配置\n"),
        ("configs/model_configs.yaml", "# 模型配置\n"),
        ("setup.py", "# 安装脚本\n"),
    ]
    
    # 确认操作
    if not args.dry_run and not args.force:
        print("将执行以下操作:")
        print("1. 创建标准目录结构")
        print("2. 移动文件到标准位置")
        print("3. 创建必要的初始化文件")
        
        confirm = input("\n确认执行这些操作? [y/N]: ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return 0
    
    # 创建目录结构
    create_dir_structure(".", standard_dirs, args.dry_run)
    
    # 移动文件
    moved_files = []
    for src, dst in file_mappings:
        if move_file(src, dst, args.dry_run):
            moved_files.append((src, dst))
    
    # 创建空文件
    for path, content in empty_files:
        create_empty_file(path, content, args.dry_run)
    
    # 清理过度嵌套的目录
    if not args.dry_run:
        nested_dirs = [
            "celldetection/models/backbone",
            "celldetection/models/detection",
            "celldetection/models/attention",
            "celldetection/models/neck",
            "core/data"
        ]
        for dir_path in nested_dirs:
            if os.path.exists(dir_path) and not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    print(f"已删除空目录: {dir_path}")
                except:
                    pass
    
    # 打印摘要
    if args.dry_run:
        print("\n模拟执行完成! 以上是将要执行的操作.")
    else:
        print("\n整理完成!")
        print(f"- 创建了 {len(standard_dirs)} 个标准目录")
        print(f"- 移动了 {len(moved_files)} 个文件")
        print(f"- 创建了 {len(empty_files)} 个初始化文件")
        
        # 提醒用户更新导入语句
        if moved_files:
            print("\n注意: 文件位置已更改，您可能需要更新导入语句。可以使用以下命令检查:")
            print("  python tools/check_migration.py --source_dir .")
    
    return 0

if __name__ == '__main__':
    exit(main()) 