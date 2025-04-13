# 酵母细胞检测项目 - 代码迁移指南

本文档提供了从旧版本代码迁移到新版本的详细指南。我们重构了部分代码，引入了更好的模块组织和命名约定。本指南将帮助您顺利完成迁移过程。

## 重要变更概述

1. 数据增强模块移动到了新位置
2. 模型实现改进并重命名
3. 优化了项目结构
4. 添加了新功能

## 导入语句更新

以下是主要的导入路径变更，你需要更新代码中的相应导入语句：

| 旧路径 | 新路径 | 说明 |
|-------|--------|------|
| `from utils.data.augmentation import ...` | `from core.data.augment import ...` | 数据增强模块重构 |
| `from models.yolov10 import ...` | `from models.yolov10_yeast import ...` | 模型专门针对酵母细胞优化 |
| `from utils.visualization import ...` | 保持不变 | 可视化模块路径未变更 |

## 代码示例

### 数据增强模块

**旧代码:**

```python
from utils.data.augmentation import RandomBrightness, RandomContrast

transform = Compose([
    RandomBrightness(0.3),
    RandomContrast(0.3)
])
```

**新代码:**

```python
from core.data.augment import RandomBrightness, RandomContrast

transform = Compose([
    RandomBrightness(0.3),
    RandomContrast(0.3)
])
```

### 模型定义

**旧代码:**

```python
from models.yolov10 import YOLOv10

model = YOLOv10(
    backbone='cspdarknet',
    num_classes=3
)
```

**新代码:**

```python
from models.yolov10_yeast import YOLOv10

model = YOLOv10(
    backbone='cspdarknet',
    num_classes=3,
    cell_attention=True  # 新增的细胞注意力机制
)
```

## API变更

### 添加的新功能

1. **细胞注意力机制**
   
   新模型中添加了专门针对细胞形态的注意力机制，提高了检测精度：
   
   ```python
   # 启用细胞注意力机制
   model = YOLOv10(cell_attention=True)
   ```

2. **出芽细胞检测模式**
   
   ```python
   # 启用出芽细胞检测模式
   python manage_data.py annotate --mode semi --bud_mode
   ```

3. **活力检测模式（美兰染色）**
   
   ```python
   # 启用细胞活力检测模式
   python manage_data.py annotate --mode semi --viability_mode
   ```

### 修改的接口

1. **训练配置**
   
   训练配置参数有所变化：
   
   ```python
   # 旧版本
   model.train(
     batch_size=16,
     learning_rate=0.001
   )
   
   # 新版本
   model.train(
     batch_size=16,
     learning_rate=0.001,
     augmentation_level='strong',  # 新增参数：增强等级
     focal_loss=True               # 新增参数：使用Focal Loss
   )
   ```

## 数据格式变更

数据格式基本保持不变，但标注文件包含了更多信息：

**旧版标注格式:**
```json
{
  "cells": [
    {"x": 100, "y": 150, "width": 30, "height": 30}
  ]
}
```

**新版标注格式:**
```json
{
  "cells": [
    {"x": 100, "y": 150, "width": 30, "height": 30, "type": "normal", "confidence": 0.95}
  ],
  "metadata": {
    "image_quality": "good",
    "magnification": "40x"
  }
}
```

## 迁移步骤

1. **备份当前代码**
   
   在进行任何更改前，请先备份您的工作：
   
   ```bash
   # 创建代码备份
   cp -r your_working_directory backup_code
   ```

2. **运行迁移检查工具**
   
   我们提供了一个工具来检查您的代码是否需要迁移：
   
   ```bash
   # 检查需要迁移的代码
   python tools/check_migration.py --source_dir your_code_dir
   ```

3. **更新导入语句**
   
   按照上述导入路径对应表更新您的导入语句。

4. **测试功能**
   
   更新后，请运行测试以确保功能正常：
   
   ```bash
   # 运行测试
   python tools/test_migration.py
   ```

## 常见问题

1. **找不到模块 'core.data.augment'**
   - 确保您已更新到最新版本
   - 检查项目结构是否正确
   - 运行 `pip install -e .` 重新安装开发模式

2. **模型加载错误**
   - 如果尝试加载旧模型权重，可能需要转换：
     ```python
     from tools.convert_weights import convert_old_to_new
     new_weights = convert_old_to_new('old_weights.pth', 'new_weights.pth')
     ```

3. **配置文件不兼容**
   - 更新配置文件中的模型定义部分：
     ```yaml
     # 旧配置
     model:
       type: YOLOv10
       backbone: cspdarknet
     
     # 新配置
     model:
       type: YOLOv10  # 类名保持不变，但会使用新的实现
       backbone: cspdarknet
       cell_attention: true  # 新参数
     ```

## 结论

迁移到新版本将获得更好的性能和新功能，包括改进的出芽细胞检测和活力分析。如果遇到任何迁移问题，请参考项目README或提交问题报告。 