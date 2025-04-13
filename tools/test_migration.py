#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='测试代码迁移是否成功')
    parser.add_argument('--test_dir', type=str, default='.',
                        help='测试目录，默认为当前目录')
    return parser.parse_args()

def test_imports():
    """测试主要模块的导入"""
    modules_to_test = [
        'core.data.augment',
        'models.yolov10_yeast'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            results[module_name] = True
            print(f"✅ 成功导入模块: {module_name}")
        except ImportError as e:
            results[module_name] = False
            print(f"❌ 导入模块失败: {module_name}")
            print(f"   错误信息: {str(e)}")
    
    return all(results.values())

def test_module_compatibility():
    """测试模块兼容性"""
    tests = [
        {
            'name': '数据增强模块',
            'code': '''
try:
    from core.data.augment import RandomBrightness, RandomContrast
    
    # 创建增强器实例
    brightness = RandomBrightness(0.3)
    contrast = RandomContrast(0.3)
    print("✅ 成功创建数据增强器实例")
except Exception as e:
    print(f"❌ 创建数据增强器失败: {str(e)}")
    raise
'''
        },
        {
            'name': '模型实例化',
            'code': '''
try:
    from models.yolov10_yeast import YOLOv10
    
    # 创建模型实例
    model = YOLOv10(backbone='cspdarknet', num_classes=3)
    print("✅ 成功创建模型实例")
except Exception as e:
    print(f"❌ 创建模型实例失败: {str(e)}")
    raise
'''
        }
    ]
    
    success = True
    for test in tests:
        print(f"\n测试: {test['name']}")
        try:
            exec(test['code'])
        except Exception as e:
            success = False
            print(f"❌ 测试 '{test['name']}' 失败")
    
    return success

def test_file_structure(test_dir):
    """测试文件结构"""
    required_paths = [
        'core/data/augment.py',
        'models/yolov10_yeast.py',
        'docs/migration_guide.md'
    ]
    
    success = True
    for path in required_paths:
        full_path = os.path.join(test_dir, path)
        if os.path.exists(full_path):
            print(f"✅ 文件存在: {path}")
        else:
            print(f"❌ 文件不存在: {path}")
            success = False
    
    return success

def main():
    args = parse_args()
    
    print("="*60)
    print("开始测试代码迁移")
    print("="*60)
    
    # 添加当前目录到搜索路径
    sys.path.insert(0, args.test_dir)
    
    # 测试导入
    print("\n[1/3] 测试模块导入")
    import_success = test_imports()
    
    # 测试模块兼容性
    print("\n[2/3] 测试模块兼容性")
    compatibility_success = test_module_compatibility()
    
    # 测试文件结构
    print("\n[3/3] 测试文件结构")
    structure_success = test_file_structure(args.test_dir)
    
    # 总结
    print("\n"+"="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"模块导入: {'✅ 通过' if import_success else '❌ 失败'}")
    print(f"模块兼容性: {'✅ 通过' if compatibility_success else '❌ 失败'}")
    print(f"文件结构: {'✅ 通过' if structure_success else '❌ 失败'}")
    
    if import_success and compatibility_success and structure_success:
        print("\n🎉 所有测试通过，代码迁移成功！")
        return 0
    else:
        print("\n⚠️ 测试未全部通过，请检查错误并修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 