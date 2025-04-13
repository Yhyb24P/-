import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class YeastDetectionLoss(nn.Module):
    """
    酵母细胞检测的损失函数
    
    结合了边界框回归、目标性和分类损失
    
    Args:
        num_classes: 类别数量
        box_weight: 边界框回归损失权重
        obj_weight: 目标性损失权重
        cls_weight: 分类损失权重
    """
    def __init__(self, num_classes: int, box_weight: float = 1.0, 
                obj_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        # 边界框回归损失使用CIoU Loss
        self.box_loss = CIoULoss()
        
        # 目标性损失使用BCE Loss
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 分类损失使用BCE Loss或CrossEntropy Loss
        if num_classes == 1:  # 二分类
            self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        else:  # 多分类
            self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
            
    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            predictions: 模型预测结果，可以是单个张量或者预测结果列表
            targets: 目标标注列表，每个元素形状为 (num_objects, 5) [class, x, y, w, h]
            
        Returns:
            包含各种损失的字典
        """
        # 确定设备，predictions可能是单个张量或者列表
        if isinstance(predictions, list):
            device = predictions[0].device
            # 简化处理：仅使用第一个特征图的预测
            pred = predictions[0]
        else:
            device = predictions.device
            pred = predictions
            
        # 处理YOLO格式 (batch, anchors, height, width, channels)
        if len(pred.shape) == 5:
            batch_size, num_anchors, height, width, _ = pred.shape
            pred = pred.reshape(batch_size, -1, pred.shape[-1])  # (batch, anchors*h*w, channels)
        else:
            batch_size = pred.shape[0]
        
        # 初始化损失
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        # 汇总批次损失
        for batch_idx in range(batch_size):
            current_pred = pred[batch_idx]  # (anchors*h*w, channels)
            current_target = targets[batch_idx]  # (num_objects, 5)
            
            # 提取预测的边界框、目标性分数和类别分数
            pred_boxes = current_pred[:, :4]  # (anchors*h*w, 4) [x, y, w, h]
            pred_obj = current_pred[:, 4]     # (anchors*h*w) 目标性分数
            pred_cls = current_pred[:, 5:]    # (anchors*h*w, num_classes) 类别分数
            
            # 如果没有目标，所有预测都是背景
            if current_target.shape[0] == 0:
                obj_loss += self.obj_loss(pred_obj, torch.zeros_like(pred_obj))
                continue
                
            # 提取目标的类别和边界框
            target_cls = current_target[:, 0].long()  # (num_objects) 类别索引
            target_boxes = current_target[:, 1:5]     # (num_objects, 4) [x, y, w, h]
            
            # 计算IoU矩阵
            try:
                ious = bbox_iou(pred_boxes, target_boxes)  # (anchors*h*w, num_objects)
            except Exception as e:
                print(f"IoU计算错误: {e}")
                print(f"pred_boxes形状: {pred_boxes.shape}, target_boxes形状: {target_boxes.shape}")
                # 返回零损失，避免训练中断
                return {
                    'box_loss': box_loss,
                    'obj_loss': obj_loss,
                    'cls_loss': cls_loss,
                    'total_loss': box_loss + obj_loss + cls_loss
                }
            
            # 为每个预测找到最匹配的目标
            best_ious, best_idxs = ious.max(dim=1)
            
            # 创建目标掩码 (IoU大于阈值的预测框被视为正样本)
            iou_threshold = 0.5
            pos_mask = best_ious > iou_threshold
            
            # 计算目标性损失
            target_obj = torch.zeros_like(pred_obj)
            if pos_mask.sum() > 0:
                target_obj[pos_mask] = 1.0
            obj_loss += self.obj_loss(pred_obj, target_obj)
            
            # 如果有正样本，计算边界框和分类损失
            if pos_mask.sum() > 0:
                try:
                    # 计算边界框回归损失（只对正样本）
                    matched_pred_boxes = pred_boxes[pos_mask]
                    matched_target_idxs = best_idxs[pos_mask]
                    matched_target_boxes = target_boxes[matched_target_idxs]
                    
                    # 确保形状正确
                    if matched_pred_boxes.shape[0] > 0 and matched_target_boxes.shape[0] > 0:
                        box_loss += self.box_loss(matched_pred_boxes, matched_target_boxes)
                    
                    # 计算分类损失（只对正样本）
                    matched_pred_cls = pred_cls[pos_mask]
                    matched_target_cls = target_cls[matched_target_idxs]
                    
                    if self.num_classes == 1:  # 二分类
                        cls_loss += self.cls_loss(matched_pred_cls.squeeze(-1), 
                                                torch.ones_like(matched_pred_cls.squeeze(-1)))
                    else:  # 多分类
                        cls_loss += self.cls_loss(matched_pred_cls, matched_target_cls)
                except Exception as e:
                    print(f"计算正样本损失时出错: {e}")
                    # 继续训练，不中断，使用现有损失
        
        # 计算平均损失
        total_loss = (self.box_weight * box_loss + 
                      self.obj_weight * obj_loss + 
                      self.cls_weight * cls_loss)
        
        return {
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
            'total_loss': total_loss
        }

def compute_loss(predictions, targets, model):
    """
    计算YOLOv10损失
    
    Args:
        predictions: 模型预测输出
        targets: 目标标注
        model: 模型对象
        
    Returns:
        总损失和各组件损失
    """
    device = predictions[0].device
    
    # 初始化损失组件
    lbox = torch.zeros(1, device=device)  # 边界框损失
    lobj = torch.zeros(1, device=device)  # 置信度损失
    lcls = torch.zeros(1, device=device)  # 分类损失
    
    # 处理各尺度的预测结果
    for i, pred in enumerate(predictions):
        # 匹配目标和计算损失
        tobj, tcls, tbox = build_targets(pred, targets, model.anchors[i])
        
        # 计算边界框损失
        if tbox.shape[0] > 0:
            pred_box = pred[tobj > 0][:, :4]
            lbox += bbox_ciou(pred_box, tbox).mean()
        
        # 计算置信度损失
        lobj += smooth_bce(pred[..., 4], tobj)
        
        # 计算分类损失
        if model.num_classes > 1:
            lcls += smooth_bce(pred[..., 5:], tcls)
    
    # 根据权重合并损失
    loss = lbox + lobj + lcls
    
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


class CIoULoss(nn.Module):
    """
    完整IoU损失（CIoU Loss）
    
    结合了IoU、中心点距离和长宽比，更好地优化边界框
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算CIoU损失
        
        Args:
            pred: 预测边界框，形状为 (n, 4) [x, y, w, h]
            target: 目标边界框，形状为 (n, 4) [x, y, w, h]
            
        Returns:
            CIoU损失
        """
        # 确保输入形状匹配且至少有一个边界框
        if pred.shape[0] == 0 or target.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # 如果形状不匹配，尝试进行广播
        if pred.shape[0] != target.shape[0]:
            # 如果目标只有一个框但预测有多个，复制目标
            if target.shape[0] == 1 and pred.shape[0] > 1:
                target = target.repeat(pred.shape[0], 1)
            # 如果预测只有一个框但目标有多个，复制预测
            elif pred.shape[0] == 1 and target.shape[0] > 1:
                pred = pred.repeat(target.shape[0], 1)
            else:
                # 其他情况无法广播，返回零损失
                print(f"警告：无法广播形状 pred={pred.shape}, target={target.shape}")
                return torch.tensor(0.0, device=pred.device)
                
        # 计算CIoU损失
        try:
            ciou_values = bbox_ciou(pred, target)
            # 转换为损失
            loss = 1 - ciou_values
            
            # 应用减少
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # 'none'
                return loss
        except Exception as e:
            print(f"CIoU计算错误: {e}")
            return torch.tensor(0.0, device=pred.device)


def bbox_iou(box1, box2, xywh=True, giou=False, diou=False, ciou=False, eps=1e-7):
    """
    计算两组边界框之间的IoU
    
    Args:
        box1: 第一组边界框，形状为 (n, 4)
        box2: 第二组边界框，形状为 (m, 4)
        xywh: 是否为[x, y, w, h]格式，否则为[x1, y1, x2, y2]
        giou: 是否计算GIoU
        diou: 是否计算DIoU
        ciou: 是否计算CIoU
        
    Returns:
        IoU矩阵，形状为 (n, m)
    """
    # 转换为xyxy格式，如果需要
    if xywh:
        box1_xyxy = xywh2xyxy(box1)
        box2_xyxy = xywh2xyxy(box2)
    else:
        box1_xyxy = box1
        box2_xyxy = box2
    
    # 确保输入是2D张量
    if box1_xyxy.dim() > 2:
        box1_xyxy = box1_xyxy.reshape(-1, 4)
    if box2_xyxy.dim() > 2:
        box2_xyxy = box2_xyxy.reshape(-1, 4)
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy[:, 0], box1_xyxy[:, 1], box1_xyxy[:, 2], box1_xyxy[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy[:, 0], box2_xyxy[:, 1], box2_xyxy[:, 2], box2_xyxy[:, 3]
    
    # 广播维度用于批量计算
    b1_x1 = b1_x1.unsqueeze(1)  # (n,1)
    b1_y1 = b1_y1.unsqueeze(1)  # (n,1)
    b1_x2 = b1_x2.unsqueeze(1)  # (n,1)
    b1_y2 = b1_y2.unsqueeze(1)  # (n,1)
    
    # 计算交集区域的坐标
    inter_x1 = torch.max(b1_x1, b2_x1)  # (n,m)
    inter_y1 = torch.max(b1_y1, b2_y1)  # (n,m)
    inter_x2 = torch.min(b1_x2, b2_x2)  # (n,m)
    inter_y2 = torch.min(b1_y2, b2_y2)  # (n,m)
    
    # 计算交集面积，确保宽高为正
    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # (n,m)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)  # (n,m)
    inter_area = inter_w * inter_h  # (n,m)
    
    # 计算两个框的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # (n,1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # (m,)
    
    # 计算并集面积和IoU
    union_area = b1_area + b2_area - inter_area  # (n,m)
    iou = inter_area / (union_area + eps)  # (n,m)
    
    return iou


def bbox_ciou(box1, box2, eps=1e-7):
    """
    计算CIoU (Complete IoU)
    
    Args:
        box1: 预测边界框 [x, y, w, h]
        box2: 目标边界框 [x, y, w, h]
        
    Returns:
        CIoU值
    """
    # 转换为xyxy格式
    box1_xyxy = xywh2xyxy(box1)
    box2_xyxy = xywh2xyxy(box2)
    
    # 计算IoU
    iou = bbox_iou(box1_xyxy, box2_xyxy, xywh=False)
    
    # 计算中心点距离
    b1_cx = (box1_xyxy[:, 0] + box1_xyxy[:, 2]) / 2
    b1_cy = (box1_xyxy[:, 1] + box1_xyxy[:, 3]) / 2
    b2_cx = (box2_xyxy[:, 0] + box2_xyxy[:, 2]) / 2
    b2_cy = (box2_xyxy[:, 1] + box2_xyxy[:, 3]) / 2
    
    center_dist = ((b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2)
    
    # 计算最小外接矩形
    enclose_x1 = torch.min(box1_xyxy[:, 0], box2_xyxy[:, 0])
    enclose_y1 = torch.min(box1_xyxy[:, 1], box2_xyxy[:, 1])
    enclose_x2 = torch.max(box1_xyxy[:, 2], box2_xyxy[:, 2])
    enclose_y2 = torch.max(box1_xyxy[:, 3], box2_xyxy[:, 3])
    
    enclose_w = (enclose_x2 - enclose_x1)
    enclose_h = (enclose_y2 - enclose_y1)
    enclose_diag = enclose_w ** 2 + enclose_h ** 2 + eps
    
    # 计算宽高比一致性
    w1, h1 = box1[:, 2], box1[:, 3]
    w2, h2 = box2[:, 2], box2[:, 3]
    
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    alpha = v / (1 - iou + v + eps)
    
    # 计算CIoU
    ciou = iou - center_dist / enclose_diag - alpha * v
    
    return ciou


def xywh2xyxy(x):
    """
    将边界框从[x, y, w, h]格式转换为[x1, y1, x2, y2]格式
    
    Args:
        x: 边界框 [x, y, w, h]，支持任意维度，最后一维必须是4
        
    Returns:
        转换后的边界框 [x1, y1, x2, y2]
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def smooth_bce(pred, target, eps=1e-7):
    """
    平滑的BCE损失
    
    Args:
        pred: 预测值
        target: 目标值
        eps: 小值防止数值问题
        
    Returns:
        平滑的BCE损失
    """
    return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


def build_targets(pred, targets, anchors):
    """
    为预测分配目标
    
    Args:
        pred: 模型预测输出
        targets: 目标标注
        anchors: 当前尺度的锚框
        
    Returns:
        目标置信度、类别和边界框
    """
    # 此函数实现根据具体模型设计而定
    # 这里提供一个简化版本的占位实现
    batch_size, _, grid_h, grid_w = pred.shape[:4]
    tobj = torch.zeros_like(pred[..., 0])
    tcls = torch.zeros_like(pred[..., 5:])
    tbox = torch.zeros_like(pred[..., :4])
    
    # 实际实现中，这里应该匹配目标与预测框，并填充tobj, tcls, tbox
    
    return tobj, tcls, tbox 