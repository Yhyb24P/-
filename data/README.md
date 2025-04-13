# 酵母细胞数据

本目录包含酵母细胞显微镜图像的原始数据。

## 目录结构

```
data/
│
├── raw/               # 原始图像数据目录
│   └── *.bmp          # 原始显微镜图像
│
└── __init__.py        # Python包初始化文件
```

## 原始数据

原始数据包含多张酵母细胞显微镜图像（.bmp格式）。这些图像是后续处理和分析的基础。

## 数据处理

如需处理这些原始图像，请使用项目根目录中的`process_data.py`脚本：

```bash
# 预处理原始图像（调整大小，划分训练/验证/测试集）
python process_data.py --mode preprocess --raw_dir data/raw --output_dir data --img_size 640

# 可视化原始图像中的细胞
python process_data.py --mode visualize --raw_dir data/raw --output_dir data

# 数据增强（生成额外的训练样本）
python process_data.py --mode augment --raw_dir data/raw --output_dir data --aug_examples 3
```

此脚本整合了所有必要的数据处理功能，是对原始脚本的简化和整合。 