"""
注意力机制模块

提供各种注意力机制实现，如通道注意力、空间注意力和CBAM等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class LightweightCBAM(nn.Module):
    """轻量级CBAM注意力模块"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_attention = LightweightChannelAttention(channels, reduction_ratio)
        # 空间注意力 - 简化版本
        self.spatial_attention = LightweightSpatialAttention()

    def forward(self, x):
        # 应用通道注意力
        x = self.channel_attention(x) * x
        # 应用空间注意力
        x = self.spatial_attention(x) * x
        return x


class LightweightChannelAttention(nn.Module):
    """轻量级通道注意力模块"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 仅使用平均池化，省略最大池化以减少计算量
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out)


class LightweightSpatialAttention(nn.Module):
    """轻量级空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        # 使用单个通道卷积代替标准卷积
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 仅使用平均池化在通道维度上
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = self.conv(avg_out)
        return self.sigmoid(out)


class YeastAttention(nn.Module):
    """酵母细胞专用注意力机制"""

    def __init__(self, in_channels):
        """
        初始化

        参数:
            in_channels: 输入通道数
        """
        super().__init__()

        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 形态注意力 - 使用深度可分离卷积提取细胞形态特征
        self.morphology_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [B, C, H, W]

        返回:
            注意力加权特征
        """
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca

        # 空间注意力
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa

        # 形态注意力 - 专门针对细胞形态特征
        ma = self.morphology_attention(x_sa)
        x_ma = x_sa * ma

        return x_ma


class ScaleAdaptiveAttention(nn.Module):
    """尺度自适应注意力模块

    根据SPDCN和SAFECount方法，集成尺度信息到注意力中，
    使模型更好地适应不同分辨率下的目标检测
    """
    def __init__(self, in_channels):
        super().__init__()
        # 尺度感知特征提取
        self.scale_aware_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)

        # 通道注意力 - 基于尺度的加权
        self.scale_pool = nn.AdaptiveAvgPool2d(1)
        self.scale_channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 局部尺度增强
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 基本特征
        identity = x

        # 尺度感知特征
        scale_feat = self.scale_aware_conv(x)

        # 通道注意力权重
        scale_weight = self.scale_channel_attention(self.scale_pool(scale_feat))

        # 应用权重
        weighted_feat = scale_feat * scale_weight

        # 局部增强
        enhanced = self.local_enhance(weighted_feat)

        # 残差连接
        return identity + enhanced