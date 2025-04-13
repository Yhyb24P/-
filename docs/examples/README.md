# 酵母细胞检测项目示例代码

本目录包含酵母细胞检测项目的各种示例代码和用例。

## 使用示例

- [基本训练](training.md) - 基本模型训练示例
- [自定义训练](custom_training.md) - 自定义训练过程
- [推理示例](inference.md) - 模型推理示例
- [模型导出](export.md) - 模型导出为不同格式
- [性能优化](optimization.md) - 模型性能优化示例

## 应用场景

- [酵母细胞计数](cell_counting.md) - 酵母细胞计数应用
- [密度估计](density_estimation.md) - 酵母细胞密度估计
- [细胞生长监测](growth_monitoring.md) - 酵母细胞生长监测

## 指南

下面是一些快速入门的代码片段：

### 模型加载与推理

```python
from models.yolov10_yeast import YOLOv10_Yeast
import torch
import cv2

# 加载模型
model = YOLOv10_Yeast(num_classes=1)
model.load_pretrained("checkpoints/best.pth")
model.eval().to("cuda")

# 准备图像
img = cv2.imread("sample.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
img = img.to("cuda")

# 推理
with torch.no_grad():
    detections = model.predict(img, conf_thresh=0.15, iou_thresh=0.45)

# 处理结果
boxes = detections[0].cpu().numpy()
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    print(f"检测到酵母细胞: 位置=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], 置信度={conf:.2f}")
```

### 自定义训练循环

```python
from models.yolov10_yeast import YOLOv10_Yeast
from utils.losses.yolo_loss import YeastDetectionLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from core.data.dataset import YeastDataset

# 创建模型和优化器
model = YOLOv10_Yeast(num_classes=1)
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = YeastDetectionLoss(num_classes=1)

# 创建数据集
train_dataset = YeastDataset("data/train", "data/annotations", is_train=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练循环
model.train()
for epoch in range(100):
    epoch_loss = 0
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss_dict = criterion(predictions, targets)
        loss = loss_dict["total_loss"]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
    
# 保存模型
model.save("checkpoints/my_model.pth")
```

更多详细示例，请参考各个示例文件。 