import os
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='酵母细胞数据管理工具')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理原始图像')
    preprocess_parser.add_argument('--raw_dir', type=str, default='data/raw', help='原始图像目录')
    preprocess_parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    preprocess_parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    preprocess_parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    preprocess_parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    preprocess_parser.add_argument('--img_size', type=int, default=512, help='图像调整大小')
    preprocess_parser.add_argument('--auto_annotate', action='store_true', help='自动生成标注')
    preprocess_parser.add_argument('--aug_intensity', type=str, choices=['mild', 'medium', 'strong'], 
                                  default='medium', help='数据增强强度')
    preprocess_parser.add_argument('--no_augmentation', action='store_true', help='禁用数据增强')
    preprocess_parser.add_argument('--aug_examples', type=int, default=0, 
                                  help='为每张图像生成的增强示例数量，0表示不生成')
    preprocess_parser.add_argument('--visualize', action='store_true', help='可视化处理结果')
    preprocess_parser.add_argument('--use_standalone', action='store_true', help='使用独立脚本避免导入问题')
    
    # 增强可视化命令
    augment_parser = subparsers.add_parser('visualize_augmentation', help='可视化数据增强效果')
    augment_parser.add_argument('--input_image', type=str, required=True, help='输入图像路径')
    augment_parser.add_argument('--num_samples', type=int, default=9, help='生成的增强样本数量')
    augment_parser.add_argument('--output_dir', type=str, default='data/augmentation_examples', help='输出目录')
    augment_parser.add_argument('--intensity', type=str, choices=['mild', 'medium', 'strong'], help='增强强度')
    
    # 标注命令
    annotate_parser = subparsers.add_parser('annotate', help='标注图像')
    annotate_parser.add_argument('--img_dir', type=str, default='data/processed', help='图像目录')
    annotate_parser.add_argument('--ann_dir', type=str, default='data/annotations', help='标注输出目录')
    annotate_parser.add_argument('--mode', type=str, choices=['auto', 'manual', 'semi'], default='semi', help='标注模式')
    annotate_parser.add_argument('--bud_mode', action='store_true', help='启用出芽细胞标注模式')
    annotate_parser.add_argument('--viability_mode', action='store_true', help='启用细胞活性检测模式（美兰染色）')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析数据集')
    analyze_parser.add_argument('--config', type=str, default='configs/train/cell_data.yaml', help='配置文件路径')
    analyze_parser.add_argument('--output_dir', type=str, default='data/stats', help='输出目录')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试数据集')
    test_parser.add_argument('--config', type=str, default='configs/train/cell_data.yaml', help='配置文件路径')
    test_parser.add_argument('--num_samples', type=int, default=5, help='显示的样本数量')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='configs/train/cell_data.yaml', help='配置文件路径')
    train_parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    train_parser.add_argument('--visualize', action='store_true', help='启用增强可视化功能')
    train_parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练')
    
    # 添加项目清理命令
    cleanup_parser = subparsers.add_parser('cleanup', help='清理项目文件结构')
    cleanup_parser.add_argument('--find_duplicates', action='store_true', help='查找重复文件')
    cleanup_parser.add_argument('--find_temp', action='store_true', help='查找临时文件')
    cleanup_parser.add_argument('--analyze', action='store_true', help='分析项目结构')
    cleanup_parser.add_argument('--clean', action='store_true', help='清理临时文件')
    cleanup_parser.add_argument('--reorganize', action='store_true', help='建议项目重组')
    cleanup_parser.add_argument('--create_structure', action='store_true', help='创建标准项目结构')
    cleanup_parser.add_argument('--all', action='store_true', help='执行所有操作')
    
    return parser.parse_args()

def run_command(command):
    """运行命令并实时输出结果"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def check_path(path, create=False):
    """检查路径是否存在，如果create=True则创建目录"""
    p = Path(path)
    if create and not p.exists():
        os.makedirs(path, exist_ok=True)
        print(f"创建目录: {path}")
    elif not p.exists():
        print(f"警告: 路径不存在: {path}")
        return False
    return True

def update_config_file(config_path, intensity=None, enabled=True):
    """更新配置文件中的增强设置"""
    try:
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 确保增强部分存在
        if 'augmentation' not in config:
            config['augmentation'] = {}
            
        # 更新配置
        config['augmentation']['enabled'] = enabled
        if intensity:
            config['augmentation']['intensity'] = intensity
            
        # 写回配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        return True
    except Exception as e:
        print(f"更新配置文件时出错: {str(e)}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    if args.command == 'preprocess':
        # 检查路径
        if not check_path(args.raw_dir):
            print("错误: 原始图像目录不存在")
            return
        
        # 确保输出目录与配置文件中的一致
        config_path = 'configs/annotation_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 使用配置文件中的路径
        processed_data_dir = config.get('paths', {}).get('processed_data', 'data/processed_images')
        visualization_dir = config.get('visualization', {}).get('output_dir', 'data/visualization')
        
        # 创建目录
        check_path(processed_data_dir, create=True)
        check_path(visualization_dir, create=True)
        
        # 更新输出目录参数
        args.output_dir = processed_data_dir
        
        # 更新配置文件中的增强设置
        if not update_config_file(
            config_path, 
            intensity=args.aug_intensity, 
            enabled=not args.no_augmentation
        ):
            print("警告: 无法更新配置文件中的增强设置，将使用默认设置")
        
        # 检查是否使用独立脚本
        if args.use_standalone:
            # 使用独立脚本，避免导入问题
            cmd = [
                f"python scripts/standalone_preprocess.py",
                f"--config {config_path}"
            ]
            
            if args.raw_dir:
                cmd.append(f"--raw_dir {args.raw_dir}")
                
            if args.visualize:
                cmd.append("--visualize")
                
            if args.aug_examples > 0:
                cmd.append(f"--aug_examples {args.aug_examples}")
                
            if args.no_augmentation:
                cmd.append("--no_augmentation")
            else:
                cmd.append(f"--aug_intensity {args.aug_intensity}")
        else:
            # 使用原始脚本
            cmd = [
                f"python scripts/preprocess_images.py",
                f"--config {config_path}"
            ]
            
            if args.visualize:
                cmd.append("--visualize")
                
            if args.aug_examples > 0:
                cmd.append(f"--aug_examples {args.aug_examples}")
        
        # 运行预处理
        print("=== 开始预处理 ===")
        print(f"使用增强强度: {args.aug_intensity if not args.no_augmentation else '无增强'}")
        print(f"使用{'独立' if args.use_standalone else '标准'}脚本")
        return run_command(" ".join(cmd))
    
    elif args.command == 'visualize_augmentation':
        # 检查路径
        if not os.path.isfile(args.input_image):
            print(f"错误: 输入图像不存在: {args.input_image}")
            return
        
        check_path(args.output_dir, create=True)
        
        # 构建命令
        cmd = [
            f"python scripts/visualize_augmentations.py",
            f"--input_image {args.input_image}",
            f"--num_samples {args.num_samples}",
            f"--output_dir {args.output_dir}"
        ]
        
        if args.intensity:
            cmd.append(f"--intensity {args.intensity}")
        
        # 运行增强可视化
        print("=== 开始生成增强示例 ===")
        return run_command(" ".join(cmd))
    
    elif args.command == 'annotate':
        # 检查路径
        if not check_path(args.img_dir):
            print("错误: 图像目录不存在")
            return
        
        check_path(args.ann_dir, create=True)
        
        # 使用默认标注工具
        # 构建命令
        cmd = [
            f"python scripts/annotate_images.py",
            f"--img_dir {args.img_dir}",
            f"--ann_dir {args.ann_dir}",
            f"--mode {args.mode}"
        ]

        # 添加出芽细胞标注模式参数
        if args.bud_mode:
            cmd.append("--bud_mode")
        
        # 添加细胞活性检测模式参数
        if args.viability_mode:
            cmd.append("--viability_mode")
        
        # 运行标注
        print("=== 开始标注 ===")
        return run_command(" ".join(cmd))
    
    elif args.command == 'analyze':
        # 检查路径
        check_path(args.output_dir, create=True)
        
        # 构建命令
        cmd = [
            f"python tools/summary_dataset.py",
            f"--config {args.config}",
            f"--output_dir {args.output_dir}"
        ]
        
        # 运行分析
        print("=== 开始分析 ===")
        return run_command(" ".join(cmd))
    
    elif args.command == 'test':
        # 构建命令
        cmd = [
            f"python tools/test_dataset.py",
            f"--config {args.config}",
            f"--num_samples {args.num_samples}"
        ]
        
        # 运行测试
        print("=== 开始测试 ===")
        return run_command(" ".join(cmd))
    
    elif args.command == 'train':
        # 构建命令
        cmd = [
            f"python train_cell.py",
            f"--config {args.config}"
        ]
        
        if args.resume:
            cmd.append(f"--resume {args.resume}")
        
        if args.visualize:
            cmd.append("--visualize")
        
        if args.amp:
            cmd.append("--amp")
        
        # 运行训练
        print("=== 开始训练 ===")
        return run_command(" ".join(cmd))
    
    elif args.command == 'cleanup':
        # 构建命令
        cmd = [
            f"python tools/cleanup_project.py",
            f"--dir ."
        ]
        
        if args.find_duplicates:
            cmd.append("--find_duplicates")
            
        if args.find_temp:
            cmd.append("--find_temp")
            
        if args.analyze:
            cmd.append("--analyze")
            
        if args.clean:
            cmd.append("--clean")
            
        if args.reorganize:
            cmd.append("--reorganize")
            
        if args.create_structure:
            cmd.append("--create_structure")
            
        if args.all:
            cmd.append("--all")
        
        # 运行项目清理
        print("=== 开始清理项目 ===")
        return run_command(" ".join(cmd))
    
    else:
        print("请选择一个命令: preprocess, visualize_augmentation, annotate, analyze, test, train")
        return 1

if __name__ == '__main__':
    sys.exit(main())