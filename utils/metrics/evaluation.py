import numpy as np
from typing import List, Dict, Any, Tuple, Union

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1: 边界框1，格式为[x, y, w, h]
        box2: 边界框2，格式为[x, y, w, h]
        
    Returns:
        IoU值
    """
    # 转换为[x1, y1, x2, y2]格式
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # 计算交集区域
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # 计算各自面积
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # 计算IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return max(0.0, min(1.0, iou))

def evaluate_detection(ground_truth: List[Dict[str, Any]], 
                     predictions: List[Dict[str, Any]], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    评估检测结果
    
    Args:
        ground_truth: 真实标注，格式为包含'bbox'的字典列表
        predictions: 预测结果，格式为包含'bbox'的字典列表
        iou_threshold: IoU阈值，用于判断是否为真阳性
        
    Returns:
        包含评估指标的字典
    """
    if not ground_truth and not predictions:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'mean_iou': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
    
    if not ground_truth:
        return {
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0,
            'mean_iou': 0.0,
            'tp': 0,
            'fp': len(predictions),
            'fn': 0
        }
    
    if not predictions:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mean_iou': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': len(ground_truth)
        }
    
    # 提取边界框
    gt_boxes = [item['bbox'] for item in ground_truth if 'bbox' in item]
    pred_boxes = [item['bbox'] for item in predictions if 'bbox' in item]
    
    # 计算所有配对的IoU
    ious = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            ious[i, j] = compute_iou(gt_box, pred_box)
    
    # 匹配
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    
    # 按IoU从大到小排序
    iou_indices = np.dstack(np.unravel_index(np.argsort(-ious.ravel()), ious.shape))[0]
    
    for idx in iou_indices:
        gt_idx, pred_idx = idx
        # 如果IoU大于阈值且两者都未匹配
        if ious[gt_idx, pred_idx] >= iou_threshold and \
           gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matched_ious.append(ious[gt_idx, pred_idx])
    
    # 计算指标
    tp = len(matched_gt)  # 真阳性：正确检测的细胞
    fp = len(pred_boxes) - tp  # 假阳性：错误检测的细胞
    fn = len(gt_boxes) - tp  # 假阴性：未检测到的细胞
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def count_cells(annotations: List[Dict[str, Any]]) -> int:
    """
    计算标注中的细胞数量
    
    Args:
        annotations: 标注数据
        
    Returns:
        细胞数量
    """
    return len([item for item in annotations if 'bbox' in item])

def calculate_metrics_from_files(ground_truth_files: List[str], 
                               prediction_files: List[str], 
                               iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    从文件计算评估指标
    
    Args:
        ground_truth_files: 真实标注文件列表
        prediction_files: 预测结果文件列表
        iou_threshold: IoU阈值
        
    Returns:
        汇总评估指标
    """
    import json
    
    all_metrics = []
    
    for gt_file, pred_file in zip(ground_truth_files, prediction_files):
        # 加载真实标注
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        # 加载预测结果
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        
        # 提取细胞列表
        gt_cells = gt_data.get('cells', [])
        pred_cells = pred_data.get('cells', [])
        
        # 评估
        metrics = evaluate_detection(gt_cells, pred_cells, iou_threshold)
        all_metrics.append(metrics)
    
    # 计算平均指标
    avg_metrics = {}
    for key in ['precision', 'recall', 'f1_score', 'mean_iou']:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # 计算总数
    avg_metrics['total_gt'] = sum(m['tp'] + m['fn'] for m in all_metrics)
    avg_metrics['total_pred'] = sum(m['tp'] + m['fp'] for m in all_metrics)
    avg_metrics['total_tp'] = sum(m['tp'] for m in all_metrics)
    avg_metrics['total_fp'] = sum(m['fp'] for m in all_metrics)
    avg_metrics['total_fn'] = sum(m['fn'] for m in all_metrics)
    
    return avg_metrics

def print_metrics_report(metrics: Dict[str, float]) -> None:
    """
    打印评估指标报告
    
    Args:
        metrics: 评估指标字典
    """
    print("=" * 50)
    print("检测评估报告")
    print("=" * 50)
    
    # 打印主要指标
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1 Score): {metrics['f1_score']:.4f}")
    print(f"平均IoU (Mean IoU): {metrics['mean_iou']:.4f}")
    
    # 打印数量指标
    if 'total_gt' in metrics:
        print("\n数量统计:")
        print(f"总真实标注数量: {metrics['total_gt']}")
        print(f"总预测数量: {metrics['total_pred']}")
        print(f"真阳性数量 (TP): {metrics['total_tp']}")
        print(f"假阳性数量 (FP): {metrics['total_fp']}")
        print(f"假阴性数量 (FN): {metrics['total_fn']}")
    else:
        print("\n数量统计:")
        print(f"真阳性数量 (TP): {metrics['tp']}")
        print(f"假阳性数量 (FP): {metrics['fp']}")
        print(f"假阴性数量 (FN): {metrics['fn']}")
    
    print("=" * 50) 