#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测训练脚本

此脚本实现了基于YOLOv10_Yeast模型的酵母细胞检测训练，
包含高分辨率图像处理、密集细胞检测和高级数据增强功能。
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path
from tqdm import tqdm

# 导入自定义模块
from models.yolov10_yeast import YOLOv10
from utils.losses import YeastDetectionLoss
from utils.metrics import calculate_metrics
from utils.postprocess import non_max_suppression, adaptive_nms
from models.backbone.backbone import YeastBackbone


class ModelEMA:
    """模型指数移动平均
    
    对模型权重进行平滑以提高测试时的稳定性
    
    Args:
        model: 要应用EMA的模型
        decay: EMA衰减率 (默认: 0.9999)
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA权重到模型进行评估"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
    def state_dict(self):
        """获取EMA的状态字典"""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup
        }
        
    def load_state_dict(self, state_dict):
        """加载EMA状态字典"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='酵母细胞与出芽细胞检测训练')
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
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的本地排名')
    return parser.parse_args()


def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config, args):
    """创建数据加载器"""
    from core.data.augment import get_yeast_cell_transforms
    from celldetection.data.dataset import YeastCellDataset

    # 确定图像尺寸
    img_size = config['data']['image_size']
    if args.high_res:
        img_size = args.high_res_size
        print(f"启用高分辨率模式，图像尺寸: {img_size}")
    
    # 获取数据目录
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']
    ann_dir = config['data']['ann_dir']
    
    # 获取数据增强参数
    augment_level = config['data'].get('augment_level', 'medium')
    
    # 创建数据增强转换
    train_transforms = get_yeast_cell_transforms(
        img_size=img_size, 
        is_train=True, 
        level=augment_level,
        include_mixup=args.efficient_train,
        include_mosaic=args.efficient_train
    )
    
    val_transforms = get_yeast_cell_transforms(
        img_size=img_size, 
        is_train=False
    )
    
    # 创建数据集
    train_dataset = YeastCellDataset(
        image_dir=train_dir,
        annotation_dir=ann_dir,
        transform=train_transforms,
        is_train=True,
        multi_scale=args.multi_scale
    )
    
    val_dataset = YeastCellDataset(
        image_dir=val_dir,
        annotation_dir=ann_dir,
        transform=val_transforms,
        is_train=False
    )
    
    # 调整批量大小（针对高分辨率和高效训练）
    batch_size = config['train']['batch_size']
    if args.batch_size is not None:
        batch_size = args.batch_size
    
    if args.high_res and not args.efficient_train:
        # 高分辨率图像需要更小的批量大小
        original_batch = batch_size
        scale_factor = (args.high_res_size / config['data']['image_size']) ** 2
        batch_size = max(1, int(batch_size / scale_factor))
        print(f"高分辨率模式下批量大小从 {original_batch} 调整为 {batch_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    return train_loader, val_loader


def build_model(config, device, high_res=False):
    """构建模型"""
    model_config = config['model']
    num_classes = model_config['num_classes']
    
    # 创建模型，当high_res=True时启用高分辨率优化
    model = YOLOv10(
        backbone=model_config.get('backbone', 'cspdarknet'),
        num_classes=num_classes,
        cell_attention=model_config.get('cell_attention', True)
    )
    
    # 预训练加载（如果配置）
    pretrained = model_config.get('pretrained', None)
    if pretrained:
        try:
            model.load_pretrained(pretrained)
            print(f"成功加载预训练权重: {pretrained}")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    # 移动到设备
    model = model.to(device)
    
    return model


def train(config, args):
    """训练模型"""
    # 设置随机种子以提高可重复性
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config, args)
    
    # 构建模型
    model = build_model(config, device, args.high_res)
    
    # 创建损失函数
    loss_config = config.get('loss', {})
    use_focal = loss_config.get('focal_loss', False)
    use_balanced_l1 = loss_config.get('balanced_l1', False)
    
    criterion = YeastDetectionLoss(
        num_classes=config['model']['num_classes'],
        box_weight=loss_config.get('box_weight', 1.0),
        obj_weight=loss_config.get('obj_weight', 1.0),
        cls_weight=loss_config.get('cls_weight', 1.0),
        focal_loss=use_focal,
        balanced_l1=use_balanced_l1
    ).to(device)
    
    # 创建优化器
    lr = config['train'].get('lr', 0.001)
    weight_decay = config['train'].get('weight_decay', 0.0005)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # 创建学习率调度器
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs'],
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            verbose=True
        )
    
    # 混合精度训练
    scaler = GradScaler()
    use_amp = args.amp
    if use_amp:
        print("启用自动混合精度训练")
    
    # EMA支持
    ema_enabled = config.get('train', {}).get('ema', {}).get('enabled', False)
    if ema_enabled:
        ema_decay = config.get('train', {}).get('ema', {}).get('decay', 0.9999)
        ema = ModelEMA(model, decay=ema_decay)
        print(f"启用EMA, 衰减率: {ema_decay}")
    else:
        ema = None
    
    # 训练配置
    train_config = config.get('train', {})
    epochs = train_config.get('epochs', 100)
    save_interval = train_config.get('save_interval', 10)
    log_interval = train_config.get('log_interval', 10)
    eval_interval = train_config.get('eval_interval', 1)
    save_dir = train_config.get('save_dir', 'weights')
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard目录
    log_dir = train_config.get('log_dir', 'runs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_map = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'best_map' in checkpoint:
                best_map = checkpoint['best_map']
            
            if 'ema' in checkpoint and ema is not None:
                ema.load_state_dict(checkpoint['ema'])
            
            print(f"已恢复到第 {start_epoch} 轮训练，最佳mAP: {best_map:.4f}")
        else:
            print(f"检查点 '{args.resume}' 不存在, 从头开始训练")
    
    print("开始训练...")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n--- 第 {epoch+1}/{epochs} 轮训练 ---")
        
        # 训练一个轮次
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            log_interval=log_interval,
            writer=writer,
            ema=ema,
            high_res=args.high_res,
            efficient_train=args.efficient_train
        )
        
        # 更新学习率调度器
        if scheduler_type == 'plateau':
            scheduler.step(train_loss)
        else:
            scheduler.step()
        
        # 定期评估
        if (epoch + 1) % eval_interval == 0:
            # 使用EMA模型进行评估
            if ema is not None:
                ema.apply_shadow()
            
            # 评估模型
            val_loss, metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                writer=writer,
                high_res=args.high_res
            )
            
            # 恢复EMA模型
            if ema is not None:
                ema.restore()
            
            # 记录指标
            mAP = metrics['mAP']
            precision = metrics['precision']
            recall = metrics['recall']
            
            writer.add_scalar('val/mAP', mAP, epoch)
            writer.add_scalar('val/precision', precision, epoch)
            writer.add_scalar('val/recall', recall, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            
            print(f"验证: Loss: {val_loss:.4f}, mAP: {mAP:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # 保存最佳模型
            if mAP > best_map:
                best_map = mAP
                save_path = os.path.join(save_dir, 'best.pth')
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_map': best_map
                }
                
                if ema is not None:
                    checkpoint['ema'] = ema.state_dict()
                
                torch.save(checkpoint, save_path)
                print(f"保存最佳模型 mAP: {best_map:.4f} 到 {save_path}")
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'checkpoint_{epoch+1}.pth')
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_map
            }
            
            if ema is not None:
                checkpoint['ema'] = ema.state_dict()
            
            torch.save(checkpoint, save_path)
            print(f"保存检查点到 {save_path}")
    
    # 训练结束
    writer.close()
    print(f"训练完成! 最佳mAP: {best_map:.4f}")


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, 
                   scaler, use_amp, log_interval, writer, ema=None,
                   high_res=False, efficient_train=False):
    """训练一个轮次"""
    model.train()
    epoch_loss = 0
    batches_processed = 0
    
    # 梯度累积步数 - 在高分辨率模式下启用梯度累积以节省内存
    accumulation_steps = 1
    if high_res and efficient_train:
        accumulation_steps = 2
        print(f"高分辨率高效训练模式: 启用梯度累积 ({accumulation_steps} 步)")
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # 将数据移到设备
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        
        # 前向传播
        with autocast(enabled=use_amp):
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            # 梯度累积
            loss = loss / accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积更新
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # 更新EMA模型
            if ema is not None:
                ema.update()
        
        # 更新统计信息
        epoch_loss += loss.item() * accumulation_steps
        batches_processed += 1
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f"{epoch_loss / batches_processed:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # 记录训练信息
        if (batch_idx + 1) % log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item() * accumulation_steps, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
    
    # 确保最后一个批次的梯度也被更新
    if len(train_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if ema is not None:
            ema.update()
    
    # 返回平均损失
    return epoch_loss / batches_processed


def validate(model, val_loader, criterion, device, epoch, writer, high_res=False):
    """验证模型"""
    model.eval()
    val_loss = 0
    
    # 收集所有预测和真实标签以计算指标
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            # 将数据移到设备
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss = criterion(predictions, targets)
            val_loss += loss.item()
            
            # 收集预测和目标
            # 这里使用适用于高分辨率图像的自适应NMS
            nms_predictions = adaptive_nms(
                predictions, 
                conf_thresh=0.05, 
                iou_thresh=0.45,
                density_aware=high_res  # 为高分辨率图像启用密度感知NMS
            )
            
            all_predictions.extend(nms_predictions)
            all_targets.extend(targets)
    
    # 计算平均损失
    val_loss /= len(val_loader)
    
    # 计算指标
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return val_loss, metrics


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置参数（如果在命令行中指定）
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    
    if args.device:
        config['device'] = args.device
    
    # 使用高分辨率模式
    if args.high_res:
        config['data']['image_size'] = args.high_res_size
    
    # 启用AMP
    if args.amp:
        config['train']['amp'] = True
    
    # 训练模型
    train(config, args)


if __name__ == "__main__":
    main() 