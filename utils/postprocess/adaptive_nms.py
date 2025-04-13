"""
自适应NMS模块

本模块提供针对密集酵母细胞检测场景的高级NMS算法:
- 自适应NMS: 基于目标密度动态调整IoU阈值
- Soft-NMS: 不直接抑制重叠框，而是降低其置信度
- 连接组件分析: 合并重叠严重的目标框
- 密度感知NMS: 使用细胞密度图调整NMS阈值
- 多阈值NMS: 使用不同置信度阈值的NMS结果组合

作者: Yeast Cell Detection Team
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union


def adaptive_nms(
    predictions: List[torch.Tensor],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    density_aware: bool = False,
    density_factor: float = 0.1,
    max_detections: int = 300
) -> List[torch.Tensor]:
    """自适应非极大值抑制
    
    根据检测区域的目标密度动态调整IoU阈值，
    在密集区域使用较小的IoU阈值，在稀疏区域使用较大的IoU阈值。
    
    Args:
        predictions: 模型预测输出列表
        conf_thresh: 置信度阈值
        iou_thresh: 基础IoU阈值
        density_aware: 是否启用密度感知模式
        density_factor: 密度影响因子
        max_detections: 每张图像的最大检测数
        
    Returns:
        经过NMS处理的检测结果列表
    """
    from utils.postprocess import non_max_suppression, xywh2xyxy
    
    # 如果没有启用密度感知，直接使用标准NMS
    if not density_aware:
        return non_max_suppression(
            predictions, 
            conf_thres=conf_thresh, 
            iou_thres=iou_thresh, 
            max_det=max_detections
        )
    
    batch_size = len(predictions[0])
    results = []
    
    for batch_idx in range(batch_size):
        # 收集该图像的所有预测
        image_preds = []
        
        for pred_level in predictions:
            pred = pred_level[batch_idx]
            
            # 处理不同形状的预测
            if len(pred.shape) == 5:  # [B, A, H, W, C]
                B, A, H, W, _ = pred.shape
                pred = pred.reshape(B * A * H * W, -1)
            elif len(pred.shape) != 2:  # 不是[N, C]
                continue
                
            # 置信度过滤
            conf_mask = pred[..., 4] > conf_thresh
            pred = pred[conf_mask]
            
            if not pred.shape[0]:
                continue
                
            # 转换为xyxy格式
            pred_boxes = xywh2xyxy(pred[..., :4])
            
            # 合并预测
            cls_conf, cls_idx = torch.max(pred[..., 5:], dim=1)
            image_preds.append(
                torch.cat([
                    pred_boxes,
                    pred[..., 4:5],  # objectness
                    cls_idx.unsqueeze(1).float(),
                    cls_conf.unsqueeze(1)
                ], dim=1)
            )
        
        if not image_preds:
            results.append(None)
            continue
            
        # 合并所有尺度的预测
        image_preds = torch.cat(image_preds, dim=0)
        
        # 按置信度排序
        conf_sort_idx = torch.argsort(image_preds[:, 4], descending=True)
        image_preds = image_preds[conf_sort_idx]
        
        # 限制检测数量
        if image_preds.shape[0] > max_detections:
            image_preds = image_preds[:max_detections]
        
        # 计算框的密度
        def calculate_density(boxes):
            """计算每个框的局部密度"""
            # 扩展框以创建局部区域
            expanded_boxes = boxes.clone()
            # 扩大区域为原来的3倍
            width = expanded_boxes[:, 2] - expanded_boxes[:, 0]
            height = expanded_boxes[:, 3] - expanded_boxes[:, 1]
            expanded_boxes[:, 0] -= width
            expanded_boxes[:, 1] -= height
            expanded_boxes[:, 2] += width
            expanded_boxes[:, 3] += height
            
            # 计算扩展框与所有框的IoU
            density = torch.zeros(boxes.shape[0], device=boxes.device)
            
            for i in range(boxes.shape[0]):
                # 计算当前扩展框与所有框的IoU
                b1 = expanded_boxes[i:i+1]
                b2 = boxes
                
                # 计算交集
                x1 = torch.max(b1[:, 0], b2[:, 0])
                y1 = torch.max(b1[:, 1], b2[:, 1])
                x2 = torch.min(b1[:, 2], b2[:, 2])
                y2 = torch.min(b1[:, 3], b2[:, 3])
                
                w = torch.clamp(x2 - x1, min=0)
                h = torch.clamp(y2 - y1, min=0)
                
                # 计算在扩展区域内的框数量
                intersect = (w * h) > 0
                density[i] = intersect.sum()
            
            return density
        
        # 计算密度
        densities = calculate_density(image_preds[:, :4])
        
        # 规范化密度到[0, 1]范围
        if densities.max() > densities.min():
            norm_densities = (densities - densities.min()) / (densities.max() - densities.min())
        else:
            norm_densities = torch.zeros_like(densities)
        
        # 执行自适应NMS
        det_max = image_preds.shape[0]
        keep = torch.zeros(det_max, dtype=torch.bool, device=image_preds.device)
        areas = (image_preds[:, 2] - image_preds[:, 0]) * (image_preds[:, 3] - image_preds[:, 1])
        
        for i in range(det_max):
            # 如果这个框已经被移除，跳过
            if not keep[i]:
                continue
                
            # 保留当前框
            keep[i] = True
            
            # 获取其它框
            order = torch.arange(det_max, device=image_preds.device)
            
            # 计算当前框和其它框的IoU
            xx1 = torch.max(image_preds[i, 0], image_preds[:, 0])
            yy1 = torch.max(image_preds[i, 1], image_preds[:, 1])
            xx2 = torch.min(image_preds[i, 2], image_preds[:, 2])
            yy2 = torch.min(image_preds[i, 3], image_preds[:, 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas - inter)
            
            # 当前框的密度
            current_density = norm_densities[i]
            
            # 动态调整IoU阈值 - 密度越高，阈值越低
            adaptive_iou_thresh = iou_thresh * (1 - density_factor * current_density)
            adaptive_iou_thresh = max(0.2, adaptive_iou_thresh)  # 设置最小阈值
            
            # 移除高IoU的框
            overlap_mask = (iou > adaptive_iou_thresh) & (order > i)
            keep[overlap_mask] = False
        
        # 保留通过NMS的框
        keep_idx = torch.where(keep)[0]
        nms_preds = image_preds[keep_idx]
        
        # 添加到结果
        results.append(nms_preds)
    
    return results


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.3,
    sigma: float = 0.5,
    score_thresh: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """软性非极大值抑制
    
    不直接抑制重叠框，而是降低其置信度，
    使用高斯惩罚函数或线性惩罚函数。
    
    Args:
        boxes: 边界框 [N, 4] (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        iou_thresh: IoU阈值
        sigma: 高斯惩罚函数的参数
        score_thresh: 分数阈值
        
    Returns:
        保留的框和它们的分数
    """
    # 复制输入以避免修改原始数据
    boxes = boxes.clone()
    scores = scores.clone()
    
    # 获取框数量
    N = boxes.shape[0]
    
    # 计算框面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 保存最终保留的框的索引
    keep = []
    
    # 按分数排序（降序）
    _, order = scores.sort(descending=True)
    
    while order.numel() > 0:
        # 选择分数最高的框
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)
        
        # 计算其余框与当前框的IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.max(torch.zeros(1, device=boxes.device), xx2 - xx1)
        h = torch.max(torch.zeros(1, device=boxes.device), yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 高斯惩罚函数
        weight = torch.exp(-(iou * iou) / sigma)
        
        # 更新分数
        scores[order[1:]] *= weight
        
        # 移除低于阈值的框
        remain_idx = torch.where(scores[order[1:]] > score_thresh)[0]
        order = order[1:][remain_idx]
    
    # 返回保留的框和更新的分数
    keep = torch.tensor(keep, device=boxes.device)
    return boxes[keep], scores[keep]


def connected_component_analysis(
    boxes: torch.Tensor,
    iou_thresh: float = 0.5
) -> torch.Tensor:
    """连接组件分析
    
    合并重叠严重的目标框，解决粘连细胞问题。
    
    Args:
        boxes: 边界框 [N, 4] (x1, y1, x2, y2)
        iou_thresh: IoU阈值
        
    Returns:
        合并后的框
    """
    if boxes.shape[0] == 0:
        return boxes
    
    # 计算IoU矩阵
    def box_iou(box1, box2):
        # 计算交集
        x1 = torch.max(box1[:, None, 0], box2[:, 0])
        y1 = torch.max(box1[:, None, 1], box2[:, 1])
        x2 = torch.min(box1[:, None, 2], box2[:, 2])
        y2 = torch.min(box1[:, None, 3], box2[:, 3])
        
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        inter = w * h
        
        # 计算各自面积
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        # 计算IoU
        iou = inter / (area1[:, None] + area2 - inter)
        return iou
    
    # 构建连接组件
    iou_matrix = box_iou(boxes, boxes)
    connected = iou_matrix > iou_thresh
    
    # 查找连接组件
    N = boxes.shape[0]
    components = torch.arange(N, device=boxes.device)
    
    # 连接组件
    for i in range(N):
        for j in range(i+1, N):
            if connected[i, j]:
                components[j] = components[i]
    
    # 合并重叠框
    merged_boxes = []
    unique_components = torch.unique(components)
    
    for comp in unique_components:
        mask = components == comp
        
        # 如果只有一个框，直接使用
        if mask.sum() == 1:
            merged_boxes.append(boxes[mask][0])
        # 如果有多个框，取并集
        else:
            comp_boxes = boxes[mask]
            x1 = torch.min(comp_boxes[:, 0])
            y1 = torch.min(comp_boxes[:, 1])
            x2 = torch.max(comp_boxes[:, 2])
            y2 = torch.max(comp_boxes[:, 3])
            merged_boxes.append(torch.tensor([x1, y1, x2, y2], device=boxes.device))
    
    # 返回合并后的框
    if merged_boxes:
        return torch.stack(merged_boxes)
    else:
        return boxes


def density_based_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    density_map: torch.Tensor,
    base_iou_thresh: float = 0.45,
    min_iou_thresh: float = 0.2
) -> torch.Tensor:
    """基于密度的NMS
    
    使用细胞密度图动态调整NMS阈值，
    密度高的区域使用更低阈值，密度低的区域使用更高阈值。
    
    Args:
        boxes: 边界框 [N, 4] (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        density_map: 密度图 [H, W]
        base_iou_thresh: 基础IoU阈值
        min_iou_thresh: 最小IoU阈值
        
    Returns:
        保留的框索引
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, device=boxes.device, dtype=torch.long)
    
    # 获取图像尺寸
    h, w = density_map.shape
    
    # 获取每个框中心的密度值
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    
    # 规范化为图像坐标
    x_indices = (centers_x * w).long().clamp(0, w - 1)
    y_indices = (centers_y * h).long().clamp(0, h - 1)
    
    # 获取密度值
    density_values = density_map[y_indices, x_indices]
    
    # 规范化密度值到[0, 1]
    if density_map.max() > density_map.min():
        normalized_density = (density_values - density_map.min()) / (density_map.max() - density_map.min())
    else:
        normalized_density = torch.zeros_like(density_values)
    
    # 调整阈值
    dynamic_thresholds = base_iou_thresh - (base_iou_thresh - min_iou_thresh) * normalized_density
    
    # 执行NMS
    keep = []
    order = torch.argsort(scores, descending=True)
    
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        # 计算IoU
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 使用当前框的阈值
        thresh = dynamic_thresholds[i]
        
        # 保留低于阈值的框
        inds = torch.where(iou <= thresh)[0]
        order = order[1:][inds]
    
    # 返回保留的框索引
    return torch.tensor(keep, device=boxes.device)


def multi_threshold_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_thresh: float = 0.5,
    conf_thresholds: List[float] = [0.05, 0.1, 0.25]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """多阈值NMS
    
    使用多个置信度阈值执行NMS，然后合并结果。
    这种方法可以获得更好的召回率，同时保持较高的精度。
    
    Args:
        boxes: 边界框 [N, 4] (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        class_ids: 类别ID [N]
        iou_thresh: IoU阈值
        conf_thresholds: 置信度阈值列表，从低到高
        
    Returns:
        合并后的框、分数和类别
    """
    from utils.postprocess import non_max_suppression
    
    # 确保阈值从低到高排序
    conf_thresholds = sorted(conf_thresholds)
    
    # 保存结果
    kept_boxes = []
    kept_scores = []
    kept_classes = []
    
    # 对每个置信度阈值执行NMS
    for conf_thresh in conf_thresholds:
        # 过滤当前阈值的框
        mask = scores >= conf_thresh
        if not mask.any():
            continue
            
        current_boxes = boxes[mask]
        current_scores = scores[mask]
        current_classes = class_ids[mask]
        
        # 执行NMS
        # 这里我们使用自己的NMS实现，但也可以使用non_max_suppression
        keep_indices = []
        order = torch.argsort(current_scores, descending=True)
        
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep_indices.append(i)
                break
                
            i = order[0].item()
            keep_indices.append(i)
            
            # 计算IoU
            xx1 = torch.max(current_boxes[i, 0], current_boxes[order[1:], 0])
            yy1 = torch.max(current_boxes[i, 1], current_boxes[order[1:], 1])
            xx2 = torch.min(current_boxes[i, 2], current_boxes[order[1:], 2])
            yy2 = torch.min(current_boxes[i, 3], current_boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            areas = (current_boxes[:, 2] - current_boxes[:, 0]) * (current_boxes[:, 3] - current_boxes[:, 1])
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = torch.where(iou <= iou_thresh)[0]
            order = order[1:][inds]
        
        # 保存结果
        keep_indices = torch.tensor(keep_indices, device=boxes.device)
        kept_boxes.append(current_boxes[keep_indices])
        kept_scores.append(current_scores[keep_indices])
        kept_classes.append(current_classes[keep_indices])
    
    # 合并结果
    if kept_boxes:
        all_boxes = torch.cat(kept_boxes, dim=0)
        all_scores = torch.cat(kept_scores, dim=0)
        all_classes = torch.cat(kept_classes, dim=0)
        
        # 再次执行NMS以移除重复框
        nms_indices = []
        order = torch.argsort(all_scores, descending=True)
        
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                nms_indices.append(i)
                break
                
            i = order[0].item()
            nms_indices.append(i)
            
            # 计算IoU
            xx1 = torch.max(all_boxes[i, 0], all_boxes[order[1:], 0])
            yy1 = torch.max(all_boxes[i, 1], all_boxes[order[1:], 1])
            xx2 = torch.min(all_boxes[i, 2], all_boxes[order[1:], 2])
            yy2 = torch.min(all_boxes[i, 3], all_boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = torch.where(iou <= iou_thresh)[0]
            order = order[1:][inds]
        
        # 最终结果
        nms_indices = torch.tensor(nms_indices, device=boxes.device)
        return all_boxes[nms_indices], all_scores[nms_indices], all_classes[nms_indices]
    else:
        # 如果没有保留的框，返回空结果
        return (torch.zeros((0, 4), device=boxes.device),
                torch.zeros(0, device=boxes.device),
                torch.zeros(0, device=boxes.device, dtype=torch.long)) 