#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测训练接口

提供模型训练的命令行接口和训练函数。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging
from pathlib import Path

# 导入自定义模块
from models.detector import YeastCellDetector
from data.datasets.cell_dataset import CellDataset
from data.utils.loss import SimpleCellLoss

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个轮次
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        
    Returns:
        平均损失
    """
    model.train()
    
    total_loss = 0
    box_loss = 0
    obj_loss = 0
    cls_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, data in enumerate(progress_bar):
        # 准备数据
        images = data['image'].to(device)
        targets = data['targets'].to(device)
        
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss_dict = criterion(predictions, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss_dict['total_loss'].item()
        box_loss += loss_dict['box_loss'].item()
        obj_loss += loss_dict['obj_loss'].item()
        cls_loss += loss_dict['cls_loss'].item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'box_loss': box_loss / (batch_idx + 1),
            'obj_loss': obj_loss / (batch_idx + 1),
            'cls_loss': cls_loss / (batch_idx + 1)
        })
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_box_loss = box_loss / len(dataloader)
    avg_obj_loss = obj_loss / len(dataloader)
    avg_cls_loss = cls_loss / len(dataloader)
    
    return {
        'total': avg_loss,
        'box': avg_box_loss,
        'obj': avg_obj_loss,
        'cls': avg_cls_loss
    }

def validate(model, dataloader, criterion, device):
    """验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        平均损失
    """
    model.eval()
    
    total_loss = 0
    box_loss = 0
    obj_loss = 0
    cls_loss = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 准备数据
            images = data['image'].to(device)
            targets = data['targets'].to(device)
            
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss_dict = criterion(predictions, targets)
            
            # 累计损失
            total_loss += loss_dict['total_loss'].item()
            box_loss += loss_dict['box_loss'].item()
            obj_loss += loss_dict['obj_loss'].item()
            cls_loss += loss_dict['cls_loss'].item()
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_box_loss = box_loss / len(dataloader)
    avg_obj_loss = obj_loss / len(dataloader)
    avg_cls_loss = cls_loss / len(dataloader)
    
    return {
        'total': avg_loss,
        'box': avg_box_loss,
        'obj': avg_obj_loss,
        'cls': avg_cls_loss
    }

def visualize_predictions(model, dataset, device, output_dir='results', num_samples=5):
    """可视化预测结果
    
    Args:
        model: 模型
        dataset: 数据集
        device: 设备
        output_dir: 输出目录
        num_samples: 样本数
    """
    model.eval()
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        data = dataset[idx]
        image = data['image'].unsqueeze(0).to(device)
        targets = data['targets']
        
        # 前向传播
        with torch.no_grad():
            predictions = model(image)[0]
        
        # 转换预测结果
        boxes = predictions[:, 0:4].cpu()
        scores = torch.sigmoid(predictions[:, 4]).cpu()
        
        # 应用阈值
        mask = scores > 0.5
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        # 应用非极大值抑制
        from torchvision.ops import nms
        keep = nms(filtered_boxes, filtered_scores, 0.5)
        filtered_boxes = filtered_boxes[keep]
        filtered_scores = filtered_scores[keep]
        
        # 加载原始图像
        orig_image = cv2.imread(data['path'])
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        # 绘制预测框
        plt.figure(figsize=(10, 10))
        plt.imshow(orig_image)
        
        height, width = orig_image.shape[:2]
        
        for box, score in zip(filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = box.numpy()
            
            # 将坐标转换回原始图像尺寸
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            # 绘制边界框
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            plt.gca().text(x1, y1, f'{score:.2f}', bbox=dict(facecolor='red', alpha=0.5))
        
        # 保存结果
        plt.axis('off')
        plt.savefig(output_dir / f'pred_{i}.png', bbox_inches='tight')
        plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='酵母细胞检测训练')
    
    # 数据相关参数
    parser.add_argument('--train-dir', type=str, default='data/datasets/train',
                      help='训练图像目录')
    parser.add_argument('--val-dir', type=str, default='data/datasets/val',
                      help='验证图像目录')
    parser.add_argument('--ann-dir', type=str, default='data/annotations/raw',
                      help='标注目录')
    parser.add_argument('--img-size', type=int, default=224,
                      help='输入图像大小')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='批次大小')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                      help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='动量')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='数据加载器的工作线程数')
    parser.add_argument('--device', type=str, default='cuda',
                      help='使用设备 (cuda/cpu)')
    
    # 模型相关参数
    parser.add_argument('--num-classes', type=int, default=1,
                      help='类别数量')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='加载检查点路径')
    
    # 输出相关参数
    parser.add_argument('--output-dir', type=str, default='weights',
                      help='输出目录')
    parser.add_argument('--log-interval', type=int, default=10,
                      help='日志间隔')
    parser.add_argument('--visualize', action='store_true',
                      help='可视化预测结果')
    
    # 增强相关参数
    parser.add_argument('--adaptive', action='store_true',
                      help='使用自适应图像增强')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        filename=output_dir / 'training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    
    # 创建数据集
    train_dataset = CellDataset(
        args.train_dir,
        args.ann_dir,
        img_size=args.img_size,
        is_train=True,
        use_adaptive_enhancement=args.adaptive
    )
    
    val_dataset = CellDataset(
        args.val_dir,
        args.ann_dir,
        img_size=args.img_size,
        is_train=False,
        use_adaptive_enhancement=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = YeastCellDetector(num_classes=args.num_classes).to(device)
    
    # 加载检查点
    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"加载检查点: {args.checkpoint}, 从第 {start_epoch} 轮开始训练")
        else:
            print(f"检查点不存在: {args.checkpoint}")
    
    # 创建损失函数
    criterion = SimpleCellLoss()
    
    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"轮次 {epoch+1}/{args.epochs}")
        
        # 训练
        train_losses = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_losses = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_losses['total'])
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses
        }, output_dir / f'checkpoint_{epoch+1}.pth')
        
        # 如果是最佳模型，保存为最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses
            }, output_dir / 'best_model.pth')
            print(f"保存最佳模型，验证损失: {val_losses['total']:.4f}")
        
        # 打印损失
        print(f"训练损失: {train_losses['total']:.4f}, 验证损失: {val_losses['total']:.4f}")
        
        # 记录日志
        logger.info(f"轮次 {epoch+1}/{args.epochs}, "
                   f"训练损失: {train_losses['total']:.4f}, "
                   f"box: {train_losses['box']:.4f}, "
                   f"obj: {train_losses['obj']:.4f}, "
                   f"cls: {train_losses['cls']:.4f}, "
                   f"验证损失: {val_losses['total']:.4f}, "
                   f"box: {val_losses['box']:.4f}, "
                   f"obj: {val_losses['obj']:.4f}, "
                   f"cls: {val_losses['cls']:.4f}")
    
    # 可视化预测结果
    if args.visualize:
        visualize_predictions(model, val_dataset, device, output_dir=output_dir / 'visualizations')
    
    print(f"训练完成. 最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"训练完成. 最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
