# 酵母细胞检测项目

本项目实现了使用YOLO架构的自动化酵母细胞检测与计数，支持高分辨率显微镜图像中的细胞实例分割和计数。

## 项目概述

酵母细胞检测是生物医学研究和工业发酵过程中的关键任务。本项目提供了完整的工具链，用于：

1. 处理原始显微镜图像
2. 标注细胞位置
3. 训练检测模型
4. 分析统计数据
5. 推理和结果可视化

## 功能特点

- **支持多种酵母细胞类型**：
  - 正常酵母细胞
  - 出芽酵母细胞（细胞分裂阶段）
  - 活力酵母细胞（透明）
  - 死亡酵母细胞（美兰染色呈蓝色）
- **细胞活力检测**（使用美兰染色：蓝色为死亡细胞，透明为活力细胞）
- **多种标注方法**：自动、手动和半自动
- **高精度识别和分类**：基于改进的YOLOv5/v10架构
- **高分辨率特征提取**：集成了高分辨率网络模块，提高小细胞检测能力
- **细胞注意力机制**：专为酵母细胞设计的注意力模块，提高检测准确率
- **密度感知检测**：针对高密度区域的细胞检测进行了优化
- **细胞分裂分析**：能够检测和分析细胞分裂状态（出芽）
- **高级图像增强**：提供针对显微镜图像的专用增强功能
- **高级数据增强**：包括MixUp、Mosaic和细胞特异性增强等
- **适应不同的光照和密度条件**
- **强大的可视化工具**：支持检测结果、真实标注比较和数据分析可视化

## 快速开始

我们提供统一的数据管理工具，简化整个工作流程：

```bash
# 设置环境
conda env create -f environment.yml
conda activate yeast_cell

# 预处理数据
python manage_data.py preprocess --raw_dir data/raw --output_dir data

# 标注图像
python manage_data.py annotate --mode semi

# 出芽细胞标注
python manage_data.py annotate --mode semi --bud_mode

# 细胞活力标注（美兰染色）
python manage_data.py annotate --mode semi --viability_mode

# 分析数据集
python manage_data.py analyze

# 测试数据集
python manage_data.py test

# 训练模型
python manage_data.py train --config configs/train/cell_data.yaml --visualize
```

## 项目结构

项目组织如下：

```
celldetection/
├── celldetection/               # 主包
│   ├── __init__.py              # 包初始化
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── backbone.py          # 特征提取网络
│   │   ├── neck.py              # FPN网络实现
│   │   ├── heads.py             # 检测头
│   │   ├── attention.py         # 注意力机制
│   │   └── high_res_module.py   # 高分辨率模块
│   ├── data/                    # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集类
│   │   ├── transforms.py        # 数据变换
│   │   ├── augment.py           # 高级数据增强
│   │   └── utils.py             # 数据工具
│   ├── enhance/                 # 图像增强
│   │   ├── __init__.py
│   │   ├── adaptive.py          # 自适应增强
│   │   ├── guided_filter.py     # 引导滤波
│   │   ├── clahe.py             # CLAHE增强
│   │   └── small_cell.py        # 小细胞增强
│   ├── utils/                   # 通用工具
│   │   ├── __init__.py
│   │   ├── metrics.py           # 评估指标
│   │   ├── visualize.py         # 可视化工具
│   │   ├── post_process.py      # 后处理函数
│   │   └── cell_division.py     # 细胞分裂检测
│   ├── configs/                 # 配置文件
│   │   ├── model_configs.yaml   # 模型配置
│   │   └── train_configs.yaml   # 训练配置
│   ├── train.py                 # 训练接口
│   ├── detect.py                # 检测接口
│   └── enhance.py               # 增强接口
├── scripts/                     # 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── detect.py                # 检测脚本
│   ├── enhance.py               # 单独增强脚本
│   └── analyze.py               # 细胞分析脚本
├── configs/                     # 配置文件目录
│   ├── default.yaml             # 默认配置
│   ├── train_configs.yaml       # 训练配置
│   └── model_configs.yaml       # 模型配置
├── setup.py                     # 安装脚本
└── README.md                    # 项目说明
```

## 数据管理

有关详细的数据管理说明，请参阅[数据处理指南](data/README.md)。

### 统一管理工具

`manage_data.py`是一个集成工具，提供以下功能：

- **预处理**：处理原始图像并分割数据集
- **标注**：支持自动、手动和半自动标注
- **分析**：生成数据集统计和可视化
- **测试**：验证数据集准备
- **训练**：启动模型训练

## 可视化工具

项目包含强大的可视化工具，位于`utils/visualization`目录：

- `visualize_cells`：可视化检测到的细胞，支持显示ID和自定义标题
- `create_cell_mask`：基于检测结果创建细胞掩码
- `create_summary_image`：创建包含原始图像、检测结果和掩码的摘要图像
- `visualize_prediction`：比较并显示预测结果与真实标注
- `visualize_detections`：可视化不同类型细胞的检测结果，用不同颜色区分

使用示例：

```python
from utils.visualization.visualization import visualize_cells, visualize_prediction

# 显示检测结果
visualize_cells(image, detected_cells, save_path="results/detection.png")

# 比较预测与真实标注
visualize_prediction(image, ground_truth, predictions,
                     save_path="results/comparison.png")
```

## 模型训练

模型训练使用YOLOv10架构，针对细胞检测任务进行了优化。

### 训练命令

```bash
# 基本训练
python train_cell.py --config configs/train/cell_data.yaml

# 启用可视化
python train_cell.py --config configs/train/cell_data.yaml --visualize

# 混合精度训练
python train_cell.py --config configs/train/cell_data.yaml --amp

# 恢复训练
python train_cell.py --config configs/train/cell_data.yaml --resume weights/checkpoint_50.pth
```

## 环境配置

项目提供两种配置环境的方式：

### 使用Conda（推荐）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate yeast_cell
```

### 使用pip

```bash
# 安装依赖
pip install -r requirements.txt
```

## 最新更新

- **框架升级与优化**：
  - 添加了高分辨率特征提取模块，提高小细胞检测能力
  - 实现了酵母细胞专用注意力机制，提高检测准确率
  - 添加了密度感知检测功能，针对高密度区域优化
  - 增强了细胞分裂分析功能，支持出芽细胞检测
  - 改进了图像增强模块，提供了更多针对显微镜图像的专用增强功能
  - 添加了高级数据增强模块，包括MixUp、Mosaic和细胞特异性增强
  - 实现了自适应NMS和Soft-NMS等高级后处理功能

- **代码重构**：
  - 推荐使用`core.data.augment`代替`utils.data.augmentation`
  - 推荐使用`models.yolov10_yeast`代替`models.yolov10`
  - 添加了迁移指南和工具，请参阅[迁移指南](docs/migration_guide.md)

- 环境配置更新：新版本支持CUDA 12.4和PyTorch 2.2.0
- 模型优化：改进了注意力机制，提高了小目标检测能力
- 项目结构优化：统一接口，删除冗余文件
- 功能增强：添加了出芽细胞和活力检测功能
- 新的环境验证工具：使用`python scripts/verify_environment.py`验证环境
- 增强了数据管理工具功能
- 改进了模型训练过程
- 优化了具有颜色编码和交互功能的可视化组件
- 重组了项目结构，提高了可维护性
- 增加了细胞可视化的配置选项
- 将实用脚本迁移到tools目录
- 清理了冗余和废弃的文件

## 冗余文件处理最佳实践

为保持项目的整洁和避免潜在的导入错误，我们推荐以下冗余文件处理流程：

1. **迁移代码依赖**
   ```bash
   # 检查需要迁移的代码
   python tools/check_migration.py --source_dir .
   ```

2. **测试迁移结果**
   ```bash
   # 测试迁移是否成功
   python tools/test_migration.py
   ```

3. **清理冗余文件**
   ```bash
   # 直接清理冗余文件（有确认提示）
   python cleanup_data.py

   # 或使用强制模式（无提示）
   python cleanup_data.py --force
   ```

4. **避免创建临时备份文件**
   - 不要创建 `.bak` 等临时备份文件
   - 如需版本控制，请使用git提交

详细的文件处理指南请参阅[任务执行指南](task.md)。

## 常见问题

1. **图像标注工具无法启动**
   - 确保已安装OpenCV（`pip install opencv-python`）
   - 在Windows上，可能需要安装Visual C++ Runtime

2. **训练期间内存错误**
   - 减小批量大小：`--batch-size 8`
   - 减小图像大小：在预处理期间设置`--img_size 512`

3. **中文路径问题**
   - 避免在文件路径中使用中文字符
   - 或将文件复制到没有中文字符的目录中

4. **找不到新模块或导入错误**
   - 项目已重构，请将旧导入语句更新为推荐的新路径：
     - 旧路径：`from utils.data.augmentation import ...`
     - 新路径：`from core.data.augment import ...`
     - 旧路径：`from models.yolov10 import ...`
     - 新路径：`from models.yolov10_yeast import ...`
   - 详细迁移指南请参考`docs/migration_guide.md`

5. **如何清理项目中的冗余文件**
   - 我们提供了专用脚本`cleanup_data.py`：
     ```bash
     # 演示模式（不删除文件）
     python cleanup_data.py --dry-run

     # 安全备份模式
     python cleanup_data.py --backup

     # 删除冗余文件
     python cleanup_data.py
     ```
   - 完整的项目结构和清理指南请参考`project_structure.md`