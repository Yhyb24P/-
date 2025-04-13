"""
形状检查工具模块

提供张量形状检查和验证功能
"""

import torch
import logging
from typing import List, Tuple, Any, Dict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prediction_shapes(predictions: Any, expected_dims: int = 5) -> bool:
    """检查预测结果的形状是否符合预期
    
    Args:
        predictions: 模型预测结果
        expected_dims: 预期的维度数量
        
    Returns:
        形状是否正确
        
    Raises:
        ValueError: 当形状不符合预期时
    """
    if not isinstance(predictions, (list, tuple)):
        logger.warning(f"不支持的预测类型: {type(predictions)}，预期为列表或元组")
        return False
        
    for i, pred in enumerate(predictions):
        if not isinstance(pred, torch.Tensor):
            logger.warning(f"预测元素 {i} 类型不正确: {type(pred)}，预期为张量")
            return False
            
        if len(pred.shape) != expected_dims:
            logger.warning(f"不支持的预测形状: {pred.shape}，预期为 {expected_dims} 维")
            return False
    
    return True

def check_target_shapes(targets: Any) -> bool:
    """检查目标标注的形状是否符合预期
    
    Args:
        targets: 目标标注列表
        
    Returns:
        形状是否正确
        
    Raises:
        ValueError: 当形状不符合预期时
    """
    if not isinstance(targets, list):
        logger.warning(f"不支持的目标类型: {type(targets)}，预期为列表")
        return False
        
    for i, target in enumerate(targets):
        if not isinstance(target, torch.Tensor):
            logger.warning(f"目标元素 {i} 类型不正确: {type(target)}，预期为张量")
            return False
            
        if len(target.shape) != 2 or target.shape[1] != 5:
            logger.warning(f"不支持的目标形状: {target.shape}，预期为 [num_objects, 5]")
            return False
    
    return True

def reshape_predictions_for_nms(predictions: List[torch.Tensor]) -> torch.Tensor:
    """重塑预测结果为NMS函数所需的形状
    
    Args:
        predictions: 模型预测结果列表
        
    Returns:
        重塑后的预测结果
    """
    # 检查预测结果类型
    if not isinstance(predictions, list):
        if isinstance(predictions, torch.Tensor) and len(predictions.shape) == 4:
            # 已经是正确的形状
            return predictions
        else:
            raise ValueError(f"不支持的预测类型: {type(predictions)}")
    
    # 假设predictions是一个包含多个特征层输出的列表
    # 我们需要将它们合并为一个单一的预测张量
    batch_size = predictions[0].shape[0]
    num_classes = predictions[0].shape[-1] - 5
    
    # 收集所有特征层的预测
    all_preds = []
    
    for pred in predictions:
        # 重塑预测结果: [batch_size, anchors, h, w, num_classes+5] -> [batch_size, anchors*h*w, num_classes+5]
        b, a, h, w, c = pred.shape
        reshaped_pred = pred.view(b, a * h * w, c)
        all_preds.append(reshaped_pred)
    
    # 合并所有预测: [batch_size, total_preds, num_classes+5]
    combined_preds = torch.cat(all_preds, dim=1)
    
    # 转换为NMS所需的格式: [batch_size, total_preds, num_classes+5]
    return combined_preds 