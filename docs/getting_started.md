# 酵母细胞检测项目入门指南

本文档提供使用YOLOv10架构的酵母细胞检测与计数项目的快速入门指南。

## 安装与环境配置

### 系统要求

- Python 3.8+
- CUDA 11.8+ (推荐CUDA 12.x)
- 16GB+ RAM (推荐)
- NVIDIA GPU with 8GB+ VRAM (推荐)

### 环境配置

#### 使用Conda (推荐)

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate yeast_cell

# 验证环境
python scripts/verify_environment.py
```

#### 使用pip

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

### 数据结构

将您的酵母细胞显微镜图像数据按以下结构组织:

```
data/
├── raw/             # 原始图像
├── processed/       # 预处理后的图像
├── annotations/     # 标注文件 (YOLO格式)
└── datasets/        # 训练数据集
    ├── train/       # 训练集图像
    ├── val/         # 验证集图像
    └── test/        # 测试集图像
```

### 数据预处理

使用预处理脚本处理原始图像：

```bash
python scripts/preprocess_images.py --source data/raw --dest data/processed
```

### 数据标注

如果您需要标注数据，可以使用以下命令启动标注工具：

```bash
python scripts/annotate.py --images data/processed --output data/annotations
```

## 模型训练

### 基本训练

使用默认配置进行训练：

```bash
python train.py --config configs/model/yolov10.yaml
```

### 启用高级功能

```bash
# 启用可视化
python train.py --config configs/model/yolov10.yaml --visualize

# 启用混合精度训练
python train.py --config configs/model/yolov10.yaml --amp

# 恢复训练
python train.py --config configs/model/yolov10.yaml --resume checkpoints/last.pth
```

### 分布式训练

```bash
# 使用两个GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train.py --config configs/model/yolov10.yaml
```

## 模型评估

### 基本评估

```bash
python scripts/evaluate.py --weights checkpoints/best.pth --data data/datasets/test
```

### 小目标评估

针对小目标酵母细胞的专用评估：

```bash
python scripts/evaluate_small_cells.py --model checkpoints/best.pth --data-dir data/datasets/test --anno-dir data/annotations
```

## 模型推理

### 单张图像推理

```bash
python inference.py --config configs/inference/default.yaml --weights checkpoints/best.pth --image path/to/image.jpg
```

### 批量推理

```bash
python inference.py --config configs/inference/default.yaml --weights checkpoints/best.pth --dir data/datasets/test
```

## TensorRT加速

### 导出模型

```bash
python scripts/export_tensorrt.py --model checkpoints/best.pth --precision fp16
```

### TensorRT推理

```bash
python inference.py --config configs/inference/tensorrt.yaml --engine exported_models/yolov10_yeast_fp16.trt --dir data/datasets/test
```

## 性能优化建议

- 对于小型酵母细胞，使用强数据增强配置：`configs/augmentation/strong.yaml`
- 降低置信度阈值（0.15-0.25）以提高小目标检测率
- 使用TensorRT进行推理加速，大型数据集上可实现2-5倍加速

## 常见问题

### 内存不足

如果遇到GPU内存不足的问题，尝试：
- 减小批量大小：`--batch-size 4`
- 减小输入图像尺寸
- 使用轻量级模型变体

### 小目标检测效果不佳

- 使用更大的输入图像尺寸（如640x640或更大）
- 启用强数据增强
- 调低置信度阈值
- 使用针对小目标优化的模型配置

## 更多资源

- [优化指南](optimization_guide.md)
- [API文档](api/README.md)
- [示例代码](examples/README.md) 