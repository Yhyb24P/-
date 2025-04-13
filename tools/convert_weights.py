#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='将旧版YOLOv10模型权重转换为新版YOLOv10_Yeast模型权重')
    parser.add_argument('--source', type=str, required=True,
                        help='源模型权重文件路径')
    parser.add_argument('--target', type=str, required=True,
                        help='目标模型权重文件路径')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细转换信息')
    return parser.parse_args()

def convert_old_to_new(source_path, target_path, verbose=False):
    """将旧版YOLOv10模型权重转换为新版YOLOv10_Yeast模型权重"""
    print(f"正在加载旧版模型权重: {source_path}")
    
    # 加载旧版权重
    if not os.path.exists(source_path):
        print(f"错误: 源权重文件不存在: {source_path}")
        return False
    
    try:
        old_weights = torch.load(source_path, map_location='cpu')
    except Exception as e:
        print(f"加载源权重时出错: {str(e)}")
        return False
    
    # 检查权重格式
    if 'model' not in old_weights and 'state_dict' not in old_weights:
        if isinstance(old_weights, dict) and any(k.startswith('backbone.') for k in old_weights.keys()):
            # 直接是状态字典
            old_state_dict = old_weights
        else:
            print("错误: 无法识别的权重格式")
            return False
    else:
        # 标准格式
        old_state_dict = old_weights.get('model', old_weights.get('state_dict'))
    
    # 创建新权重字典
    new_state_dict = {}
    
    # 映射层名称
    key_mapping = {
        # 主干网络保持不变
        'backbone.': 'backbone.',
        # FPN/PAN调整
        'neck.lateral_conv': 'neck.lateral_conv',
        'neck.fpn_conv': 'neck.fpn_conv',
        'neck.pan_conv': 'neck.pan_conv',
        # 头部映射
        'detection_head.cls_preds': 'detection_head.cls_preds',
        'detection_head.reg_preds': 'detection_head.reg_preds',
        'detection_head.obj_preds': 'detection_head.obj_preds',
    }
    
    # 转换参数
    converted_keys = set()
    skipped_keys = set()
    
    for old_key, param in old_state_dict.items():
        mapped = False
        
        for old_prefix, new_prefix in key_mapping.items():
            if old_key.startswith(old_prefix):
                # 替换前缀
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                new_state_dict[new_key] = param
                converted_keys.add(old_key)
                mapped = True
                if verbose:
                    print(f"转换: {old_key} -> {new_key}")
                break
        
        if not mapped:
            # 如果没有特定映射，保持原样
            new_state_dict[old_key] = param
            converted_keys.add(old_key)
            if verbose:
                print(f"保留: {old_key}")
    
    # 检查是否有未转换的键
    remaining_keys = set(old_state_dict.keys()) - converted_keys
    if remaining_keys and verbose:
        print("\n未转换的键:")
        for key in sorted(remaining_keys):
            print(f"  - {key}")
            skipped_keys.add(key)
    
    # 添加新的注意力模块参数（使用预训练值或随机值）
    # 新模型特有的层会在加载时自动初始化，这里无需额外处理
    
    # 保存新权重
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    new_weights = {
        'model': new_state_dict,
        'optimizer': None,
        'epoch': 0,
        'converted_from': source_path,
        'conversion_info': {
            'converted_keys': len(converted_keys),
            'skipped_keys': len(skipped_keys)
        }
    }
    
    try:
        torch.save(new_weights, target_path)
        print(f"已成功将权重保存至: {target_path}")
        print(f"统计信息: 转换 {len(converted_keys)} 项参数, 跳过 {len(skipped_keys)} 项参数")
        return True
    except Exception as e:
        print(f"保存权重时出错: {str(e)}")
        return False

def main():
    args = parse_args()
    success = convert_old_to_new(args.source, args.target, args.verbose)
    return 0 if success else 1

if __name__ == "__main__":
    main() 