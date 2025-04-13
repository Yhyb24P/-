import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DetectionHead(nn.Module):
    """YOLOv10检测头"""
    def __init__(self, num_classes, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        
        # 轻量化特征处理 - 使用深度可分离卷积
        self.feat_conv = DepthwiseSeparableConv(256, 256)
        
        # 分类头 - 轻量化
        self.cls_head = nn.Sequential(
            DepthwiseSeparableConv(256, 128),
            nn.Conv2d(128, anchors_per_scale * num_classes, 1)
        )
        
        # 边界框头
        self.box_head = nn.Sequential(
            DepthwiseSeparableConv(256, 128),
            nn.Conv2d(128, anchors_per_scale * 4, 1)
        )
        
        # 置信度头
        self.conf_head = nn.Sequential(
            DepthwiseSeparableConv(256, 64),
            nn.Conv2d(64, anchors_per_scale, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            预测结果 [B, anchors_per_scale, H, W, 5 + num_classes]
        """
        # 特征处理
        feat = self.feat_conv(x)
        
        # 分类、边界框和置信度预测
        cls_pred = self.cls_head(feat)
        box_pred = self.box_head(feat)
        conf_pred = self.conf_head(feat)
        
        # 重塑预测结果
        B, _, H, W = cls_pred.shape
        cls_pred = cls_pred.view(B, self.anchors_per_scale, self.num_classes, H, W).permute(0, 1, 3, 4, 2)
        box_pred = box_pred.view(B, self.anchors_per_scale, 4, H, W).permute(0, 1, 3, 4, 2)
        conf_pred = conf_pred.view(B, self.anchors_per_scale, 1, H, W).permute(0, 1, 3, 4, 2)
        
        # 合并预测结果
        pred = torch.cat([box_pred, conf_pred, cls_pred], dim=-1)
        
        # 应用激活函数
        pred[..., 4:5] = torch.sigmoid(pred[..., 4:5])  # 置信度
        pred[..., 5:] = torch.sigmoid(pred[..., 5:])   # 类别预测
        
        return pred

class FastFeatureProcessing(nn.Module):
    """快速特征处理模块 - 用于高分辨率图像
    
    参考TasselNetV2+方法，优化高分辨率图像处理效率
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 轻量级特征处理
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        return self.process(x)

class AdaptiveRegressorHead(nn.Module):
    """自适应回归头
    
    适应不同大小和分辨率的目标
    """
    def __init__(self, in_channels, out_channels, anchors_per_scale=3):
        super().__init__()
        # 特征处理
        self.process = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, anchors_per_scale * out_channels, 1)
        )
        
    def forward(self, x):
        return self.process(x)

class YOLOv10Head(nn.Module):
    """YOLOv10多尺度检测头"""
    def __init__(self, num_classes, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        
        # 特征融合层 - 改用轻量级设计，适配不同通道数的特征图
        self.fusion = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(256, 128),  # x3: 256通道
                DepthwiseSeparableConv(128, 256)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(512, 128),  # x4: 512通道
                DepthwiseSeparableConv(128, 256)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(1024, 128),  # x5: 1024通道
                DepthwiseSeparableConv(128, 256)
            )
        ])
        
        # 检测头
        self.heads = nn.ModuleList([
            DetectionHead(num_classes, anchors_per_scale) for _ in range(3)
        ])
        
        # 高分辨率图像快速处理模块
        self.hires_processors = nn.ModuleList([
            FastFeatureProcessing(256, 256),
            FastFeatureProcessing(512, 256),
            FastFeatureProcessing(1024, 256)
        ])
        
        # 更高效的尺度自适应检测头
        self.adaptive_box_heads = nn.ModuleList([
            AdaptiveRegressorHead(256, 4, anchors_per_scale) for _ in range(3)
        ])
        
        # 支持双标签分配策略
        self.use_dual_assignment = True
        
        # 多尺度特征增强 - 使用空洞卷积
        self.dilation_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=2, dilation=2, groups=4),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1)
            ) for _ in range(3)
        ])
        
    def forward(self, features):
        """前向传播
        
        Args:
            features: 多尺度特征图列表 [x3, x4, x5]
            
        Returns:
            多尺度预测结果列表
        """
        predictions = []
        is_high_res = features[0].shape[2] > 80  # 判断是否为高分辨率图像
        
        # 处理每个尺度的特征
        for i, (feat, fusion, head, dilation, hires_proc, adaptive_box) in enumerate(
            zip(features, self.fusion, self.heads, self.dilation_enhance, 
                self.hires_processors, self.adaptive_box_heads)):
            
            # 特征融合
            x = fusion(feat)
            
            # 高分辨率图像的特殊处理
            if is_high_res:
                # 快速特征处理 - 提高效率
                x_fast = hires_proc(x)
                
                # 空洞卷积增强 - 保持感受野
                x = x + dilation(x)
                
                # 预测
                pred = head(x)
                
                # 使用自适应回归头改进边界框预测
                box_pred = adaptive_box(x_fast)
                
                # 重塑预测结果
                B, _, H, W = pred.shape
                tmp_pred = pred.clone()
                
                # 替换边界框预测部分
                box_part = box_pred.view(B, self.anchors_per_scale, 4, H, W).permute(0, 1, 3, 4, 2)
                cls_conf_part = tmp_pred[..., 4:].view(B, self.anchors_per_scale, H, W, -1)
                
                # 合并预测结果
                pred = torch.cat([box_part, cls_conf_part], dim=-1)
                
            else:
                # 标准预测
                pred = head(x)
                
            predictions.append(pred)
            
        return predictions 