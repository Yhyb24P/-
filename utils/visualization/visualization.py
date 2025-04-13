import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

def visualize_cells(image: np.ndarray, 
                   cells: List[Union[Tuple[int, int, int, int], Dict[str, Any]]],
                   save_path: Union[str, Path] = None,
                   show_id: bool = True,
                   title: str = None) -> None:
    """
    可视化酵母细胞检测结果
    
    Args:
        image: 输入图像
        cells: 检测到的细胞列表，可以是边界框(x,y,w,h)列表或标注字典列表
        save_path: 可视化结果保存路径，若为None则显示
        show_id: 是否显示细胞ID
        title: 图像标题
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    for i, cell in enumerate(cells):
        # 处理不同类型的输入
        if isinstance(cell, tuple):
            x, y, w, h = cell
            cell_id = i + 1
        elif isinstance(cell, dict):
            if 'bbox' in cell:
                x, y, w, h = cell['bbox']
            else:
                continue
            cell_id = cell.get('id', i + 1)
        else:
            continue
        
        # 绘制边界框
        rect = plt.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1.5
        )
        plt.gca().add_patch(rect)
        
        # 显示ID
        if show_id:
            plt.text(x, y-5, f"#{cell_id}", color='red', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7, pad=0))
    
    # 设置标题
    if title:
        plt.title(title)
    
    plt.axis('off')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_cell_mask(image_shape: Tuple[int, int], 
                    cells: List[Union[Tuple[int, int, int, int], Dict[str, Any]]]) -> np.ndarray:
    """
    创建细胞掩码
    
    Args:
        image_shape: 图像形状 (H, W)
        cells: 检测到的细胞列表
        
    Returns:
        掩码图像
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for cell in cells:
        # 处理不同类型的输入
        if isinstance(cell, tuple):
            x, y, w, h = cell
        elif isinstance(cell, dict):
            if 'bbox' in cell:
                x, y, w, h = cell['bbox']
            else:
                continue
        else:
            continue
        
        # 在掩码上绘制矩形
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    return mask

def create_summary_image(image: np.ndarray, 
                        cells: List[Union[Tuple[int, int, int, int], Dict[str, Any]]],
                        save_path: Union[str, Path] = None) -> np.ndarray:
    """
    创建包含原图、检测结果和掩码的汇总图像
    
    Args:
        image: 输入图像
        cells: 检测到的细胞列表
        save_path: 保存路径
        
    Returns:
        汇总图像
    """
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 创建细胞掩码
    mask = create_cell_mask(image.shape[:2], cells)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # 创建可视化图像
    vis_image = image.copy()
    
    for cell in cells:
        # 处理不同类型的输入
        if isinstance(cell, tuple):
            x, y, w, h = cell
        elif isinstance(cell, dict):
            if 'bbox' in cell:
                x, y, w, h = cell['bbox']
            else:
                continue
        else:
            continue
        
        # 绘制矩形
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 水平连接三张图像
    summary = np.hstack((image, vis_image, mask_colored))
    
    # 保存
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    
    return summary

def visualize_prediction(image: np.ndarray, 
                       ground_truth: List[Dict[str, Any]],
                       predictions: List[Dict[str, Any]],
                       save_path: Union[str, Path] = None,
                       title: str = None) -> None:
    """
    可视化预测结果与真实标注的对比
    
    Args:
        image: 输入图像
        ground_truth: 真实标注
        predictions: 预测结果
        save_path: 保存路径
        title: 图像标题
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # 绘制真实标注
    for cell in ground_truth:
        if 'bbox' in cell:
            x, y, w, h = cell['bbox']
            rect = plt.Rectangle(
                (x, y), w, h, fill=False, edgecolor='green', linewidth=1.5,
                label='Ground Truth'
            )
            plt.gca().add_patch(rect)
    
    # 绘制预测结果
    for cell in predictions:
        if 'bbox' in cell:
            x, y, w, h = cell['bbox']
            rect = plt.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1.5,
                linestyle='--', label='Prediction'
            )
            plt.gca().add_patch(rect)
    
    # 添加标题
    if title:
        plt.title(title)
    else:
        plt.title(f"Prediction vs Ground Truth (GT: {len(ground_truth)}, Pred: {len(predictions)})")
    
    # 去除重复的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.axis('off')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def visualize_detections(image: np.ndarray, 
                         detections: List[Dict[str, Any]] = None,
                         is_target: bool = False,
                         save_path: Union[str, Path] = None,
                         title: str = None) -> np.ndarray:
    """
    可视化目标检测结果
    
    Args:
        image: 输入图像 [H,W,C]
        detections: 检测结果列表，每个包含bbox和class_id
        is_target: 是否为真实标签（用于区分颜色）
        save_path: 保存路径
        title: 图像标题
        
    Returns:
        可视化结果图像
    """
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 复制图像以免修改原图
    vis_image = image.copy()
    
    # 定义类别颜色
    colors = {
        0: (0, 255, 0),    # 普通酵母细胞 - 绿色
        1: (255, 0, 0),    # 出芽酵母细胞 - 红色
        2: (0, 0, 255),    # 活性酵母细胞 - 蓝色
        3: (255, 255, 0),  # 死亡酵母细胞 - 黄色
    }
    
    # 边框颜色
    border_color = (0, 255, 0) if is_target else (0, 0, 255)  # 真值为绿色，预测为蓝色
    
    # 绘制检测框
    if detections is not None:
        for det in detections:
            if 'bbox' in det:
                # YOLO格式 [x_center, y_center, width, height] 在0-1范围内
                x_center, y_center, width, height = det['bbox']
                
                # 转换到图像坐标
                h, w = image.shape[:2]
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 获取类别
                class_id = det.get('class_id', 0)
                color = colors.get(class_id, border_color)
                
                # 绘制边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # 添加类别和置信度标签
                conf = det.get('confidence', None)
                label = f"Class: {class_id}"
                if conf is not None:
                    label += f" {conf:.2f}"
                    
                cv2.putText(vis_image, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 添加标题
    if title:
        cv2.putText(vis_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 保存图像
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

def visualize_detection(image: np.ndarray, 
                       cells: List[Tuple[int, int, int, int]],
                       save_path: Union[str, Path] = None,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2,
                       show_count: bool = True) -> np.ndarray:
    """
    使用OpenCV可视化细胞检测结果
    
    Args:
        image: 输入图像 (RGB格式)
        cells: 检测到的细胞列表，每个元素为 (x, y, w, h) 格式的边界框
        save_path: 可视化结果保存路径，若为None则不保存
        color: 边界框颜色，默认为绿色 (0,255,0)
        thickness: 边界框线条粗细
        show_count: 是否在图像上显示检测到的细胞数量
        
    Returns:
        可视化后的图像
    """
    # 复制图像以免修改原图
    vis_image = image.copy()
    
    # 绘制检测框
    for i, (x, y, w, h) in enumerate(cells):
        # 绘制矩形
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # 显示ID
        cv2.putText(vis_image, f"#{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1, cv2.LINE_AA)
    
    # 显示总数
    if show_count:
        count_text = f"细胞数量: {len(cells)}"
        cv2.putText(vis_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    
    # 保存结果
    if save_path:
        # 确保目录存在
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转为BGR并保存
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image 