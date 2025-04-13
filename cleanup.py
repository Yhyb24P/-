#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清理冗余文件脚本

该脚本将冗余文件移动到指定的备份目录，而不是直接删除。
这样可以确保在需要时能够恢复这些文件。
"""

import os
import shutil
from pathlib import Path
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='清理项目中的冗余文件')
    parser.add_argument('--backup-dir', type=str, default='redundant_backup',
                      help='备份目录路径，默认为项目根目录下的redundant_backup')
    parser.add_argument('--dry-run', action='store_true',
                      help='只显示将要移动的文件，不实际执行')
    parser.add_argument('--delete', action='store_true',
                      help='直接删除文件而不是移动到备份目录')
    return parser.parse_args()

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_to_backup(src_path, backup_dir, dry_run=False, delete=False):
    """移动文件或目录到备份目录
    
    Args:
        src_path: 源文件或目录路径
        backup_dir: 备份目录路径
        dry_run: 是否只是演示，不实际执行
        delete: 是否直接删除而不是移动到备份
    """
    if not os.path.exists(src_path):
        print(f"[跳过] {src_path} 不存在")
        return
    
    if delete:
        action = "删除"
        if dry_run:
            print(f"[演示] 将 {action}: {src_path}")
        else:
            print(f"[执行] {action}: {src_path}")
            if os.path.isdir(src_path):
                shutil.rmtree(src_path)
            else:
                os.remove(src_path)
    else:
        # 在备份目录中创建相同的目录结构
        rel_path = os.path.relpath(src_path)
        backup_path = os.path.join(backup_dir, rel_path)
        
        # 确保目标目录存在
        backup_parent = os.path.dirname(backup_path)
        if not dry_run and not os.path.exists(backup_parent):
            os.makedirs(backup_parent)
        
        action = "移动"
        if dry_run:
            print(f"[演示] 将 {action}: {src_path} -> {backup_path}")
        else:
            print(f"[执行] {action}: {src_path} -> {backup_path}")
            if os.path.exists(backup_path):
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
            shutil.move(src_path, backup_path)

def main():
    args = parse_args()
    
    # 备份目录路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{args.backup_dir}_{timestamp}"
    
    if not args.delete and not args.dry_run:
        ensure_dir(backup_dir)
    
    # 需要清理的冗余文件和目录列表
    redundant_paths = [
        # 原始备份文件
        "backup",
        "trainer.py",
        "inference.py",
        "train_cell.py",
        
        # 冗余的工具目录
        "utils/image_processing",
        
        # 冗余的运行记录和备份
        "runs_backup",
        
        # 已迁移的旧代码
        "core",  # 旧的核心代码目录
        "yeast_cell",  # 旧的项目目录
        
        # 临时文件
        "__pycache__",  # Python缓存文件
        "simple_cell_weights"  # 旧的权重文件目录
    ]
    
    for path in redundant_paths:
        move_to_backup(path, backup_dir, args.dry_run, args.delete)
    
    # 检查其他可能的冗余文件和目录
    other_paths = [
        # 可能的临时目录和文件
        "cell_detection_results",
        "manage_data.py",
    ]
    
    print("\n以下文件/目录可能是冗余的，但建议手动确认后再清理:")
    for path in other_paths:
        if os.path.exists(path):
            print(f"  - {path}")
    
    if not args.dry_run and not args.delete:
        print(f"\n所有冗余文件已移动到: {backup_dir}")
        print("如果需要恢复，可以从该目录中复制回来。")
    elif args.dry_run:
        print("\n这只是演示模式，未执行实际操作。")
        print("如需执行实际清理，请移除 --dry-run 参数。")
    else:
        print("\n所有冗余文件已删除。")

if __name__ == "__main__":
    main() 