# 酵母细胞检测项目结构梳理

本文档清晰梳理了酵母细胞检测项目的核心结构，并提供了冗余文件清理指南。

## 项目核心结构

```
celldetection/
├── celldetection/               # 主包
│   ├── models/                  # 模型定义
│   │   ├── backbone.py          # 特征提取网络
│   │   ├── neck.py              # FPN网络实现
│   │   ├── heads.py             # 检测头
│   │   ├── attention.py         # 注意力机制
│   │   └── yolov10_yeast.py     # 推荐使用的新模型（替代yolov10.py）
│   ├── core/                    # 核心功能
│   │   ├── data/
│   │   │   └── augment.py       # 推荐使用的新增强模块（替代utils/data/augmentation.py）
│   ├── data/                    # 数据处理
│   │   ├── dataset.py           # 数据集类
│   │   ├── transforms.py        # 数据增强
│   │   └── utils.py             # 数据工具
│   ├── enhance/                 # 图像增强
│   │   ├── adaptive.py          # 自适应增强
│   │   ├── guided_filter.py     # 引导滤波
│   │   ├── clahe.py             # CLAHE增强 
│   │   └── small_cell.py        # 小细胞增强
│   ├── utils/                   # 通用工具
│   │   ├── metrics.py           # 评估指标
│   │   ├── visualize.py         # 可视化工具
│   │   └── post_process.py      # 后处理函数
│   ├── train.py                 # 训练接口
│   ├── detect.py                # 检测接口
│   └── enhance.py               # 增强接口
├── scripts/                     # 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── detect.py                # 检测脚本
│   └── enhance.py               # 单独增强脚本
├── tools/                       # 实用工具脚本（最新更新中提到）
├── configs/                     # 配置文件目录
│   ├── default.yaml             # 默认配置
│   ├── train_configs.yaml       # 训练配置
│   └── model_configs.yaml       # 模型配置
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   ├── annotations/             # 标注数据
│   ├── datasets/                # 训练集和测试集
│   └── visualization/           # 可视化结果
├── docs/                        # 文档目录
│   └── migration_guide.md       # 迁移指南
├── setup.py                     # 安装脚本
├── environment.yml              # Conda环境配置
├── requirements.txt             # Pip依赖配置
├── manage_data.py               # 统一数据管理工具
├── cleanup_data.py              # 冗余文件清理工具
├── task.md                      # 任务执行指南
├── project_structure.md         # 项目结构梳理文档
└── README.md                    # 项目说明
```

## 冗余文件清理指南

根据项目最新更新（记录在README.md中），以下文件已被替代为新的实现：

1. `utils/data/augmentation.py` → 已被 `core.data.augment` 替代
2. `models/yolov10.py` → 已被 `models.yolov10_yeast` 替代

### 清理步骤

1. **安全清理方法**

   我们提供了一个专用脚本 `cleanup_data.py` 来安全地清理冗余文件。该脚本提供了三种运行模式：

   ```bash
   # 演示模式 - 只显示将要删除的文件，不实际删除
   python cleanup_data.py --dry-run

   # 备份模式 - 将文件移至备份目录而不直接删除
   python cleanup_data.py --backup

   # 直接删除模式 - 删除冗余文件（谨慎使用）
   python cleanup_data.py
   ```

2. **推荐的清理流程**

   a. 先运行演示模式确认要删除的文件
   b. 运行备份模式进行安全备份
   c. 确认备份成功后，可以选择运行直接删除模式

3. **手动清理（如果脚本不可用）**

   如果因任何原因无法使用清理脚本，可以手动删除以下文件：
   
   ```
   utils/data/augmentation.py
   models/yolov10.py
   ```

   还应检查项目中的 `.bak` 文件，这些通常是临时备份文件。

## 代码迁移指南

为了使用新的推荐模块，需要更新代码中的导入语句：

1. 将 `from utils.data.augmentation import ...` 改为 `from core.data.augment import ...`
2. 将 `from models.yolov10 import ...` 改为 `from models.yolov10_yeast import ...`

更详细的迁移说明请参考 `docs/migration_guide.md`。

## 注意事项

1. 在清理冗余文件之前，请确保您已经了解了项目的最新结构
2. 始终先使用备份模式或演示模式，再执行实际删除操作
3. 如果您对某个文件的用途不确定，请不要删除它
4. 删除文件后可能需要更新相关导入语句
5. 如果遇到问题，可以从备份目录恢复文件

## 附录：最近更新的文件

根据README.md中的最新更新，以下改动已经应用到项目中：

1. 代码重构完成，引入新的推荐模块路径
2. 环境配置更新支持CUDA 12.4和PyTorch 2.2.0
3. 模型优化改进了注意力机制
4. 增强了数据管理工具功能
5. 优化了可视化组件
6. 将实用脚本迁移到tools目录 