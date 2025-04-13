#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
from colorama import init, Fore, Style

init()  # 初始化colorama

def parse_args():
    parser = argparse.ArgumentParser(description='检查代码中需要迁移的导入语句')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='要检查的代码目录')
    return parser.parse_args()

def check_file(file_path):
    """检查单个文件中需要迁移的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义需要迁移的导入模式
    migration_patterns = [
        {
            'pattern': r'from\s+utils\.data\.augmentation\s+import',
            'recommendation': 'from core.data.augment import',
            'description': '数据增强模块已迁移到core.data.augment'
        },
        {
            'pattern': r'from\s+models\.yolov10\s+import',
            'recommendation': 'from models.yolov10_yeast import',
            'description': '模型已优化为酵母细胞专用版本'
        },
        {
            'pattern': r'import\s+utils\.data\.augmentation',
            'recommendation': 'import core.data.augment',
            'description': '数据增强模块已迁移到core.data.augment'
        },
        {
            'pattern': r'import\s+models\.yolov10',
            'recommendation': 'import models.yolov10_yeast',
            'description': '模型已优化为酵母细胞专用版本'
        }
    ]
    
    issues = []
    for pattern_info in migration_patterns:
        matches = re.findall(pattern_info['pattern'], content)
        if matches:
            for match in matches:
                issues.append({
                    'matched_text': match,
                    'recommendation': pattern_info['recommendation'],
                    'description': pattern_info['description']
                })
    
    return issues

def check_directory(directory):
    """递归检查目录中所有Python文件"""
    all_issues = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                
                issues = check_file(file_path)
                if issues:
                    all_issues[rel_path] = issues
    
    return all_issues

def print_issues(all_issues):
    """打印发现的问题"""
    if not all_issues:
        print(f"{Fore.GREEN}✓ 没有发现需要迁移的代码!{Style.RESET_ALL}")
        return
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    total_files = len(all_issues)
    
    print(f"{Fore.YELLOW}⚠ 发现 {total_issues} 处需要迁移的代码，分布在 {total_files} 个文件中{Style.RESET_ALL}\n")
    
    for file_path, issues in all_issues.items():
        print(f"{Fore.CYAN}文件: {file_path}{Style.RESET_ALL}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {Fore.RED}{issue['matched_text']}{Style.RESET_ALL}")
            print(f"     {Fore.GREEN}推荐: {issue['recommendation']}{Style.RESET_ALL}")
            print(f"     {issue['description']}")
        print()
    
    print(f"{Fore.YELLOW}建议：将旧导入语句替换为推荐的新导入语句，详细信息请参阅 docs/migration_guide.md{Style.RESET_ALL}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.source_dir):
        print(f"{Fore.RED}错误: 指定的源目录不存在!{Style.RESET_ALL}")
        return 1
    
    print(f"正在检查目录: {args.source_dir}")
    all_issues = check_directory(args.source_dir)
    print_issues(all_issues)
    
    return 0

if __name__ == "__main__":
    main() 