# 项目结构文档

## 更新日期：2023-08-01

## 项目组织

本项目是一个用于酵母细胞检测的深度学习方案，基于改进的YOLOv10架构。它的主要目的是提供一个端到端的解决方案，用于准确检测显微镜图像中的酵母细胞。

### 主要目录结构

```
cell/
├── configs/              # 配置文件
│   ├── model/            # 模型相关配置
│   ├── train/            # 训练相关配置
│   ├── inference/        # 推理相关配置
│   ├── augmentation/     # 数据增强配置
│   └── base/             # 基础配置
├── core/                 # 核心功能模块
│   ├── model/            # 模型相关核心功能
│   └── data/             # 数据相关核心功能
├── data/                 # 数据集
│   ├── train/            # 训练数据
│   ├── val/              # 验证数据
│   └── test/             # 测试数据
├── docs/                 # 文档
├── models/               # 模型定义
│   ├── attention/        # 注意力机制
│   ├── backbone/         # 骨干网络
│   └── detection/        # 检测头
├── scripts/              # 辅助脚本
├── tools/                # 工具脚本
├── utils/                # 实用工具
│   ├── data/             # 数据处理工具
│   ├── image_processing/ # 图像处理工具
│   ├── losses/           # 损失函数
│   ├── metrics/          # 评估指标
│   ├── nms.py            # 非极大值抑制
│   ├── shape_utils.py    # 形状处理工具
│   ├── postprocess.py    # 后处理函数
│   └── visualization/    # 可视化工具
├── weights/              # 模型权重
├── train_cell.py         # 训练入口脚本
├── trainer.py            # 训练器实现
├── inference.py          # 推理脚本
└── requirements.txt      # 项目依赖
```

## 主要模块说明

### 1. 训练模块

- `train_cell.py`: 训练入口脚本，解析命令行参数并启动训练流程。
- `trainer.py`: 实现了 `YeastDetectionTrainer` 类，负责模型的训练、验证和保存。

### 2. 模型模块

- `models/yolov10_yeast.py`: 针对酵母细胞检测优化的YOLOv10模型。
- `models/attention/`: 包含不同的注意力机制实现。
- `models/backbone/`: 包含模型的骨干网络。
- `models/detection/`: 包含检测头的实现。

### 3. 工具模块

- `utils/losses/`: 包含损失函数的实现，如 `detection_loss.py`。
- `utils/metrics/`: 包含评估指标的实现，如 `detection_metrics.py`, `iou.py`, `map.py`。
- `utils/nms.py`: 改进的非极大值抑制函数。
- `utils/shape_utils.py`: 提供形状检查和重塑功能。

### 4. 配置模块

- `configs/model/yolov10.yaml`: 模型配置文件，定义了模型的架构、超参数等。
- `configs/train/`: 训练配置文件，定义了训练参数、优化器、学习率等。
- `configs/inference/`: 推理配置文件，定义了推理参数、阈值等。

## 主要执行流程

### 训练流程

1. 解析命令行参数（`train_cell.py`）。
2. 加载配置文件（`ModelBuilder.load_config`）。
3. 创建训练器实例（`YeastDetectionTrainer`）。
4. 构建模型（`ModelBuilder.build_model`）。
5. 设置损失函数、优化器和学习率调度器。
6. 执行训练循环（`trainer.train`）。
7. 在每个epoch结束后进行验证（`trainer.validate`）。
8. 根据验证结果保存模型。

### 推理流程

1. 加载配置和模型权重。
2. 预处理输入图像。
3. 执行模型前向传播。
4. 应用非极大值抑制（`non_max_suppression`）。
5. 后处理检测结果。
6. 可视化或保存结果。

## 依赖管理

项目使用以下文件管理依赖：

- `requirements.txt`: 使用pip安装的Python依赖。
- `environment.yml`: Conda环境定义。

## 数据格式

训练数据应按以下格式组织：

```
data/
├── train/
│   ├── images/         # 训练图像
│   └── labels/         # 训练标签（YOLO格式）
├── val/
│   ├── images/         # 验证图像
│   └── labels/         # 验证标签（YOLO格式）
└── test/
    ├── images/         # 测试图像
    └── labels/         # 测试标签（YOLO格式）
```

标签文件应采用YOLO格式，每行表示一个目标：`class_id x_center y_center width height`，其中坐标值已归一化到[0,1]范围。 