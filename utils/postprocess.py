"""
后处理工具模块

本模块提供目标检测模型的后处理功能:
- 边界框格式转换
- 非极大值抑制 (NMS)
- 边界框缩放和调整
- 检测结果可视化
- 细胞分裂检测
- 自适应NMS

注意: 该模块合并了之前nms.py的功能和postprocess/目录下的功能
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

# 从子模块导入高级功能
try:
    from utils.postprocess.adaptive_nms import (
        adaptive_nms,
        soft_nms,
        connected_component_analysis,
        density_based_nms,
        multi_threshold_nms
    )
    
    from utils.postprocess.cell_division import (
        detect_budding_cells,
        segment_cell_components,
        detect_viability,
        analyze_cell_growth_phase
    )
except ImportError:
    logger.warning("无法导入高级后处理功能，某些功能可能不可用")
    
    # 提供占位函数以防导入失败
    def adaptive_nms(*args, **kwargs):
        logger.error("adaptive_nms未定义")
        return None
        
    def detect_budding_cells(*args, **kwargs):
        logger.error("detect_budding_cells未定义")
        return None


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    将边界框从 [x, y, w, h] 格式转换为 [x1, y1, x2, y2] 格式
    
    Args:
        x: 边界框张量 [N, 4] 格式为 [x_center, y_center, width, height]
        
    Returns:
        [x1, y1, x2, y2] 格式的边界框
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y + h/2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """
    将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, w, h] 格式
    
    Args:
        x: 边界框张量 [N, 4] 格式为 [x1, y1, x2, y2]
        
    Returns:
        [x_center, y_center, width, height] 格式的边界框
    """
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x = (x1 + x2) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y = (y1 + y2) / 2
    y[..., 2] = x[..., 2] - x[..., 0]  # w = x2 - x1
    y[..., 3] = x[..., 3] - x[..., 1]  # h = y2 - y1
    return y


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框之间的IoU
    
    Args:
        box1: 第一组边界框 [N, 4] 格式为 [x1, y1, x2, y2]
        box2: 第二组边界框 [M, 4] 格式为 [x1, y1, x2, y2]
        
    Returns:
        IoU矩阵 [N, M]
    """
    # 扩展维度以便广播
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unsqueeze(1).chunk(4, 2)  # [N, 1, 1]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unsqueeze(0).chunk(4, 2)  # [1, M, 1]
    
    # 计算交集区域的左上角和右下角
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # 计算交集面积
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # 计算各自的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # 计算并集面积
    union_area = b1_area + b2_area - inter_area
    
    # 计算IoU
    iou = inter_area / (union_area + 1e-16)
    
    return iou


def non_max_suppression(
    predictions: List[torch.Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300
) -> List[Optional[torch.Tensor]]:
    """
    执行非极大值抑制(NMS)，处理边界框，过滤低置信度的预测
    
    Args:
        predictions: YOLOv10模型的预测输出
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        classes: 保留的类别列表，如果为None则保留所有类别
        max_det: 每张图像的最大检测框数量
        
    Returns:
        经过NMS后的检测结果列表
    """
    bs = len(predictions[0])  # 批次大小
    nc = predictions[0].shape[-1] - 5  # 类别数量(除去x,y,w,h,conf)
    
    # 初始化结果列表
    output = [None] * bs
    
    # 对每个图像处理
    for batch_idx in range(bs):
        # 收集并处理该图像的预测
        batch_predictions = []
        
        # 处理不同尺度的预测
        for preds in predictions:
            # 提取批次中当前图像的预测
            pred = preds[batch_idx]
            
            # 检查并处理预测形状
            # 支持两种形状：
            # 1. [B, A, H, W, C] - 5D 张量（原始yolo输出）
            # 2. [N, C] - 2D 张量（已处理的预测）
            
            # 重塑和转换
            if len(pred.shape) == 5:  # [B, A, H, W, C]
                B, A, H, W, _ = pred.shape
                pred = pred.reshape(B * A * H * W, -1)
            elif len(pred.shape) == 2:  # [N, C]
                # 已经处理过，直接使用
                pass
            else:
                print(f"警告: 不支持的预测形状: {pred.shape}，跳过")
                continue
            
            # 应用置信度阈值
            conf_mask = pred[..., 4] > conf_thres
            pred = pred[conf_mask]
            
            # 如果没有检测，跳过
            if not pred.shape[0]:
                continue
                
            # [x, y, w, h, conf, cls1, cls2, ...]
            pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf
            
            # 转换 bboxes 为 [x1, y1, x2, y2]
            box = xywh2xyxy(pred[..., :4])
            
            # 使用最大的类别置信度
            cls_conf, cls_idx = pred[..., 5:].max(1, keepdim=True)
            
            # 过滤指定类别
            if classes is not None:
                class_mask = torch.tensor([cls.item() in classes for cls in cls_idx.squeeze(-1)])
                if not class_mask.any():
                    continue
                pred = pred[class_mask]
                box = box[class_mask]
                cls_conf = cls_conf[class_mask]
                cls_idx = cls_idx[class_mask]
            
            # 将检测结果格式化为 [x1, y1, x2, y2, conf, cls_id]
            detections = torch.cat([box, cls_conf, cls_idx.float()], 1)
            batch_predictions.append(detections)
            
        # 如果没有检测，跳过
        if len(batch_predictions) == 0:
            continue
            
        # 合并所有尺度的预测
        detections = torch.cat(batch_predictions, 0)
        
        # 执行NMS
        # 按类别分组进行NMS
        output[batch_idx] = []
        unique_labels = detections[:, 5].unique()
        
        for cls in unique_labels:
            # 该类别的检测
            cls_dets = detections[detections[:, 5] == cls]
            
            # 按照置信度排序
            conf_sort_idx = cls_dets[:, 4].argsort(descending=True)
            cls_dets = cls_dets[conf_sort_idx]
            
            # 应用NMS
            keep = []
            while cls_dets.shape[0]:
                keep.append(cls_dets[0])
                if len(cls_dets) == 1:
                    break
                
                # 计算IoU
                ious = box_iou(cls_dets[0:1, :4], cls_dets[1:, :4]).squeeze()
                
                # 过滤重叠框
                masked_dets = cls_dets[1:][ious <= iou_thres]
                if not masked_dets.shape[0]:
                    break
                cls_dets = torch.cat([cls_dets[0:1], masked_dets])
            
            # 保存结果
            if keep:
                output[batch_idx].append(torch.stack(keep))
        
        # 合并该图像的所有检测
        if len(output[batch_idx]):
            output[batch_idx] = torch.cat(output[batch_idx], 0)
            
            # 限制检测框数量
            if output[batch_idx].shape[0] > max_det:
                conf_sort_idx = output[batch_idx][:, 4].argsort(descending=True)
                output[batch_idx] = output[batch_idx][conf_sort_idx[:max_det]]
        else:
            output[batch_idx] = None
    
    return output


def scale_boxes(boxes: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
    """
    将预测框从模型输入尺寸缩放到原始图像尺寸
    
    Args:
        boxes: 检测框 [N, 4] 格式为 [x1, y1, x2, y2]
        orig_shape: 原始图像形状 (height, width)
        
    Returns:
        缩放后的检测框
    """
    # 获取比例
    if boxes.device != 'cpu':
        orig_h, orig_w = torch.tensor(orig_shape, device=boxes.device)
    else:
        orig_h, orig_w = orig_shape
    
    # 计算当前(模型输入)尺寸
    curr_h, curr_w = boxes.shape[2:4]
    
    # 计算缩放比例
    ratio_h = orig_h / curr_h
    ratio_w = orig_w / curr_w
    
    # 应用缩放
    scaled_boxes = boxes.clone()
    scaled_boxes[..., [0, 2]] *= ratio_w  # x1, x2
    scaled_boxes[..., [1, 3]] *= ratio_h  # y1, y2
    
    return scaled_boxes


def draw_boxes(image: np.ndarray, 
              boxes: torch.Tensor, 
              classes: List[str] = None, 
              colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    在图像上绘制检测框
    
    Args:
        image: 原始图像
        boxes: 检测框 [N, 6] 格式为 [x1, y1, x2, y2, conf, cls_id]
        classes: 类别名称列表
        colors: 颜色列表
        
    Returns:
        绘制了检测框的图像
    """
    import cv2
    
    # 复制图像以避免修改原始图像
    img = image.copy()
    
    # 如果没有提供类别名称，创建默认名称
    if classes is None:
        if boxes.shape[0] > 0:
            num_classes = int(boxes[:, 5].max().item()) + 1
            classes = [f'class_{i}' for i in range(num_classes)]
        else:
            classes = ['class_0']
    
    # 如果没有提供颜色，创建默认颜色
    if colors is None:
        np.random.seed(42)
        colors = [(np.random.randint(0, 255), 
                  np.random.randint(0, 255), 
                  np.random.randint(0, 255)) for _ in range(len(classes))]
    
    # 绘制每个检测框
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box.cpu().numpy()
        cls_id = int(cls_id)
        
        # 确保索引在范围内
        cls_id = min(cls_id, len(classes) - 1)
        color = colors[cls_id]
        
        # 绘制矩形
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 添加标签
        label = f'{classes[cls_id]} {conf:.2f}'
        font_size = max(0.5, min(2, (x2 - x1) / 100))
        cv2.putText(img, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
    
    return img 