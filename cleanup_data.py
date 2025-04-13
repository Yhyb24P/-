#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清理脚本，用于删除项目中的冗余文件和备份文件
"""

import os
import shutil
import argparse
import datetime
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='清理酵母细胞检测项目中的冗余文件')
    parser.add_argument('--dry-run', action='store_true', help='仅显示要删除的文件，不实际删除')
    parser.add_argument('--backup', action='store_true', help='将文件移至备份目录而非直接删除（不推荐）')
    parser.add_argument('--force', action='store_true', help='强制删除，不提示确认')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 定义冗余文件列表
    redundant_files = [
        "utils/data/augmentation.py",  # 推荐使用 core.data.augment 代替
        "models/yolov10.py",  # 推荐使用 models.yolov10_yeast 代替
    ]
    
    # 查找所有备份文件
    backup_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".bak"):
                backup_files.append(os.path.join(root, file))
    
    # 合并所有要处理的文件
    all_files = redundant_files + backup_files
    
    # 确认操作
    if not args.dry_run and not args.force:
        print("以下文件将被删除:")
        for file_path in all_files:
            if os.path.exists(file_path):
                print(f"  - {file_path}")
        
        confirm = input("\n确认删除这些文件? [y/N]: ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return 0
    
    # 创建备份目录（如果使用备份模式）
    backup_dir = None
    if args.backup:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backup_{timestamp}")
        backup_dir.mkdir(exist_ok=True)
        print(f"已创建备份目录: {backup_dir}")
    
    # 处理文件
    deleted_count = 0
    skipped_count = 0
    for file_path in all_files:
        if not os.path.exists(file_path):
            skipped_count += 1
            if args.dry_run:
                print(f"跳过不存在的文件: {file_path}")
            continue
            
        if args.dry_run:
            print(f"将删除: {file_path}")
            deleted_count += 1
        elif args.backup:
            # 创建目标目录结构
            target_dir = backup_dir / os.path.dirname(file_path)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动文件到备份目录
            target_path = backup_dir / file_path
            shutil.move(file_path, target_path)
            print(f"已备份: {file_path} -> {target_path}")
            deleted_count += 1
        else:
            os.remove(file_path)
            print(f"已删除: {file_path}")
            deleted_count += 1
    
    # 清理空目录
    if not args.dry_run:
        empty_dirs = 0
        for root, dirs, files in os.walk(".", topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    if args.backup:
                        print(f"跳过空目录 (不备份): {dir_path}")
                    else:
                        os.rmdir(dir_path)
                        print(f"已删除空目录: {dir_path}")
                        empty_dirs += 1

        if empty_dirs > 0:
            print(f"已删除 {empty_dirs} 个空目录")
            
    # 打印清理摘要
    if args.dry_run:
        print(f"\n模拟清理完成! 将删除 {deleted_count} 个文件，跳过 {skipped_count} 个不存在的文件")
    else:
        mode = "备份" if args.backup else "删除"
        print(f"\n清理完成! 已{mode} {deleted_count} 个文件，跳过 {skipped_count} 个不存在的文件")
        if args.backup:
            print(f"所有文件已备份到: {backup_dir}")
    
    return 0

if __name__ == '__main__':
    exit(main()) 