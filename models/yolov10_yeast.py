#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
酵母细胞检测专用YOLOv10模型
用于替代旧版的models.yolov10模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.backbone import YeastBackbone
from models.attention import YeastAttention
from models.detection.head import YOLOv10Head

class YOLOv10(nn.Module):
    """YOLOv10模型，针对酵母细胞检测进行了优化"""
    
    def __init__(self, backbone='cspdarknet', num_classes=3, cell_attention=False):
        """
        初始化YOLOv10模型
        
        参数:
            backbone: 特征提取网络类型
            num_classes: 类别数量
            cell_attention: 是否使用细胞注意力机制
        """
        super().__init__()
        self.backbone_type = backbone
        self.num_classes = num_classes
        self.use_cell_attention = cell_attention
        
        # 创建主干网络
        self.backbone = self._create_backbone()
        
        # 创建颈部网络(FPN/PAN)
        self.neck = self._create_neck()
        
        # 创建检测头
        self.detection_head = self._create_detection_head()
        
        # 可选的细胞注意力模块
        self.cell_attention = None
        if cell_attention:
            self.cell_attention = self._create_cell_attention()
        
        # 注意力模块 - 为每个特征图级别创建单独的注意力模块
        self.attention_p3 = YeastAttention(256)  # 第一个特征图 (x3)
        self.attention_p4 = YeastAttention(512)  # 第二个特征图 (x4)
        self.attention_p5 = YeastAttention(1024) # 第三个特征图 (x5)
        
        # 小目标检测层
        self.small_obj_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        
        # 添加特征金字塔网络 - 从高层特征向低层特征传递语义信息
        self.fpn_p5_to_p4 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.fpn_p4_to_p3 = nn.Conv2d(512, 256, 1, 1, 0)
        
        # 添加多尺度特征融合模块
        self.scale_fusion = nn.ModuleDict({
            'p3_fusion': nn.Conv2d(256, 256, 3, padding=1),
            'p4_fusion': nn.Conv2d(512, 512, 3, padding=1),
            'p5_fusion': nn.Conv2d(1024, 1024, 3, padding=1)
        })
        
        # 支持高分辨率处理的自适应通道压缩模块
        self.hires_processors = nn.ModuleDict({
            'p3': nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(128, 256, 1)
            ),
            'p4': nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(256, 512, 1)
            ),
            'p5': nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(512, 1024, 1)
            )
        })
        
        # 无NMS训练配置
        self.no_nms_train = True
        self.topk_candidates = 100
        self.score_threshold = 0.05
        
    def _create_backbone(self):
        """创建主干网络"""
        # 实际项目中这里会有更复杂的实现
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
    
    def _create_neck(self):
        """创建颈部网络"""
        # 简化的FPN/PAN实现
        return nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
    
    def _create_detection_head(self):
        """创建检测头"""
        # 简化的检测头实现
        return nn.ModuleDict({
            'cls_preds': nn.Conv2d(256, self.num_classes, kernel_size=1),
            'reg_preds': nn.Conv2d(256, 4, kernel_size=1),
            'obj_preds': nn.Conv2d(256, 1, kernel_size=1)
        })
    
    def _create_cell_attention(self):
        """创建细胞注意力模块"""
        # 细胞形态注意力机制 - 这是对酵母细胞检测的优化
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播"""
        # 主干网络
        feat = self.backbone(x)
        
        # 颈部网络
        feat = self.neck(feat)
        
        # 应用细胞注意力（如果启用）
        if self.use_cell_attention and self.cell_attention is not None:
            attention = self.cell_attention(feat)
            feat = feat * attention
        
        # 检测头
        cls_pred = self.detection_head['cls_preds'](feat)
        reg_pred = self.detection_head['reg_preds'](feat)
        obj_pred = self.detection_head['obj_preds'](feat)
        
        return cls_pred, reg_pred, obj_pred
    
    def train(self, batch_size=16, learning_rate=0.001, 
             augmentation_level='strong', focal_loss=False):
        """
        训练模型
        
        参数:
            batch_size: 批量大小
            learning_rate: 学习率
            augmentation_level: 增强级别，'none', 'light', 'medium', 'strong'
            focal_loss: 是否使用Focal Loss
        """
        print(f"训练模型 (batch_size={batch_size}, lr={learning_rate})")
        print(f"增强级别: {augmentation_level}")
        print(f"使用Focal Loss: {focal_loss}")
        # 实际实现中这里会有完整的训练逻辑
    
    def load_pretrained(self, pretrained_path):
        """加载预训练权重"""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
            
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes
        }, path)
        
    @torch.no_grad()
    def predict(self, x, conf_thresh=0.5, iou_thresh=0.45):
        """推理预测
        
        Args:
            x: 输入图像 [B, 3, H, W]
            conf_thresh: 置信度阈值
            iou_thresh: IoU阈值
            
        Returns:
            检测结果列表 [每张图像的检测结果]
        """
        self.eval()
        
        # 获取预测
        cls_pred, reg_pred, obj_pred = self(x)
        
        # 使用双阶段TopK筛选替代NMS
        if self.no_nms_train:
            return self._topk_selection([cls_pred, reg_pred, obj_pred], conf_thresh, topk=self.topk_candidates)
            
        # 传统NMS后处理
        detections = []
        for pred in [cls_pred, reg_pred, obj_pred]:
            # 重塑预测结果
            B, A, H, W, _ = pred.shape
            pred = pred.view(B, A * H * W, -1)
            
            # 应用置信度阈值
            mask = pred[..., 4] > conf_thresh
            pred = pred[mask]
            
            if len(pred) == 0:
                detections.append(None)
                continue
                
            # 非极大值抑制
            boxes = pred[..., :4]
            scores = pred[..., 4]
            classes = pred[..., 5:].argmax(dim=-1)
            
            # 按类别分组进行NMS
            unique_classes = classes.unique()
            final_detections = []
            
            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                
                # 按分数排序
                _, order = cls_scores.sort(descending=True)
                cls_boxes = cls_boxes[order]
                cls_scores = cls_scores[order]
                
                # NMS
                keep = []
                while len(cls_boxes) > 0:
                    keep.append(0)
                    if len(cls_boxes) == 1:
                        break
                        
                    ious = self._bbox_iou(cls_boxes[0:1], cls_boxes[1:])
                    mask = ious <= iou_thresh
                    cls_boxes = cls_boxes[1:][mask]
                    cls_scores = cls_scores[1:][mask]
                    
                # 保存结果
                if keep:
                    final_detections.append(torch.cat([
                        cls_boxes[keep],
                        cls_scores[keep].unsqueeze(1),
                        torch.full((len(keep), 1), cls, device=x.device)
                    ], dim=1))
                    
            if final_detections:
                detections.append(torch.cat(final_detections))
            else:
                detections.append(None)
                
        return detections
    
    def _topk_selection(self, predictions, score_thresh=0.05, topk=100):
        """使用TopK筛选替代NMS，适用于小目标密集场景
        
        Args:
            predictions: 模型预测结果
            score_thresh: 置信度阈值
            topk: 保留的最大检测框数量
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 处理批次中的每张图像
        for batch_idx in range(predictions[0].shape[0]):
            # 合并多尺度预测结果
            pred_boxes = []
            pred_scores = []
            pred_classes = []
            
            for pred in predictions:
                # 获取当前图像的预测
                p = pred[batch_idx]
                # 展平所有尺度、所有anchor的预测
                p = p.reshape(-1, p.shape[-1])
                
                # 筛选高于阈值的预测
                mask = p[..., 4] > score_thresh
                if not mask.any():
                    continue
                    
                p = p[mask]
                
                # 收集检测框、分数和类别
                boxes = p[..., :4]
                scores = p[..., 4]
                cls_scores = p[..., 5:]
                classes = cls_scores.argmax(dim=-1)
                
                pred_boxes.append(boxes)
                pred_scores.append(scores)
                pred_classes.append(classes)
                
            # 如果没有有效预测，返回None
            if not pred_boxes:
                results.append(None)
                continue
                
            # 合并所有尺度的预测
            pred_boxes = torch.cat(pred_boxes)
            pred_scores = torch.cat(pred_scores)
            pred_classes = torch.cat(pred_classes)
            
            # 对所有预测进行TopK筛选
            if len(pred_scores) > topk:
                topk_idx = torch.topk(pred_scores, topk)[1]
                pred_boxes = pred_boxes[topk_idx]
                pred_scores = pred_scores[topk_idx]
                pred_classes = pred_classes[topk_idx]
            
            # 组织检测结果
            detections = torch.cat([
                pred_boxes,
                pred_scores.unsqueeze(-1),
                pred_classes.float().unsqueeze(-1)
            ], dim=-1)
            
            results.append(detections)
            
        return results
    
    def _bbox_iou(self, box1, box2):
        """计算IoU"""
        # 获取边界框坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
        
        # 计算交集区域
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # 计算交集面积
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集面积
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-16)
        
        return iou 