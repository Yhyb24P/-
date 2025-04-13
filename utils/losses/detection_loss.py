"""
检测损失函数模块

提供目标检测相关的损失函数实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class DetectionLoss(nn.Module):
    """目标检测损失函数
    
    包含以下损失组件：
    - 分类损失 (Classification Loss)
    - 回归损失 (Regression Loss)
    - 置信度损失 (Confidence Loss)
    """
    
    def __init__(self, num_classes: int, anchors: List[Tuple[float, float]]):
        """初始化检测损失函数
        
        Args:
            num_classes: 类别数量
            anchors: 锚框列表，每个元素为 (width, height)
        """
        super().__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        
        # 损失权重
        self.cls_weight = 1.0
        self.obj_weight = 1.0
        self.box_weight = 1.0
        
        # 损失函数
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算检测损失
        
        Args:
            predictions: 模型预测结果列表，每个元素形状为 [batch_size, num_anchors, grid_h, grid_w, num_classes + 5]
            targets: 目标标注列表，每个元素形状为 [num_objects, 5]，每行为 [class_id, x_center, y_center, width, height]
            
        Returns:
            包含各项损失的字典
        """
        device = predictions[0].device
        total_loss = 0
        loss_dict = {}
        
        # 处理每个特征层
        for pred in predictions:
            batch_size = pred.shape[0]
            grid_h, grid_w = pred.shape[2:4]
            
            # 重塑预测结果
            pred = pred.view(batch_size, -1, grid_h * grid_w, self.num_classes + 5)
            
            # 分离预测结果
            pred_obj = pred[..., 4]  # 置信度
            pred_cls = pred[..., 5:]  # 类别
            pred_box = pred[..., :4]  # 边界框
            
            # 计算目标掩码
            obj_mask = torch.zeros_like(pred_obj)
            cls_mask = torch.zeros_like(pred_cls)
            box_mask = torch.zeros_like(pred_box)
            
            # 对每个批次
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    continue
                    
                # 获取当前批次的目标
                target = targets[b]
                
                # 计算网格索引
                grid_x = (target[:, 1] * grid_w).long()
                grid_y = (target[:, 2] * grid_h).long()
                
                # 确保索引在有效范围内
                grid_x = torch.clamp(grid_x, 0, grid_w - 1)
                grid_y = torch.clamp(grid_y, 0, grid_h - 1)
                
                # 更新掩码
                for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
                    idx = gy * grid_w + gx
                    obj_mask[b, :, idx] = 1
                    cls_mask[b, :, idx, target[i, 0].long()] = 1
                    box_mask[b, :, idx] = target[i, 1:5]
            
            # 计算损失
            obj_loss = self.bce(pred_obj, obj_mask) * self.obj_weight
            cls_loss = self.bce(pred_cls, cls_mask) * self.cls_weight
            box_loss = self.mse(pred_box * obj_mask, box_mask) * self.box_weight
            
            # 累加损失
            total_loss += obj_loss + cls_loss + box_loss
            loss_dict['obj_loss'] = obj_loss
            loss_dict['cls_loss'] = cls_loss
            loss_dict['box_loss'] = box_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict 

def balanced_l1_loss(pred: torch.Tensor, 
                    target: torch.Tensor, 
                    beta: float = 1.0, 
                    alpha: float = 0.5,
                    gamma: float = 1.5,
                    reduction: str = 'mean') -> torch.Tensor:
    """
    Balanced L1损失函数
    
    在小目标检测中更有效，针对显微镜下的细胞检测优化
    
    Args:
        pred: 预测值
        target: 目标值
        beta: 平衡参数，控制小误差和大误差的权重
        alpha: 调整小目标权重的参数
        gamma: 聚焦参数
        reduction: 归约方式，'mean'，'sum'或'none'
        
    Returns:
        损失值
    """
    diff = torch.abs(pred - target)
    
    # 计算目标大小 (宽×高)，用于动态调整权重
    if len(target.shape) == 2 and target.shape[1] >= 4:  # [x,y,w,h]格式
        # 提取宽高
        target_sizes = target[:, 2] * target[:, 3]
        
        # 根据目标大小计算权重，小目标得到更高权重
        size_weight = torch.exp(-alpha * target_sizes)
        size_weight = size_weight.unsqueeze(1) if len(diff.shape) > 1 else size_weight
    else:
        size_weight = 1.0
    
    # balanced_l1 项
    b = diff + beta
    loss = torch.where(
        diff < beta,
        diff - 0.5 * beta,
        beta * torch.log(b / beta)
    )
    
    # 应用目标大小权重
    loss = loss * size_weight
    
    # 聚焦项，进一步增强对困难样本的关注
    if gamma != 1.0:
        prob = torch.exp(-loss)
        loss = torch.pow(1 - prob, gamma) * loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for对类别不平衡问题
    
    Args:
        alpha: 类别权重，可平衡正负样本
        gamma: 聚焦参数，降低易分样本的损失
        reduction: 归约方式
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-BCE_loss)
        
        # 应用focal loss公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss 