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

class PSA(nn.Module):
    """部分自注意力模块"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduced_channels = in_channels // reduction_ratio
        
        # 通道降维
        self.conv_reduce = nn.Conv2d(in_channels, self.reduced_channels, 1)
        
        # 自注意力部分
        self.query = nn.Conv2d(self.reduced_channels, self.reduced_channels, 1)
        self.key = nn.Conv2d(self.reduced_channels, self.reduced_channels, 1)
        self.value = nn.Conv2d(self.reduced_channels, self.reduced_channels, 1)
        
        # 通道恢复
        self.conv_expand = nn.Conv2d(self.reduced_channels, in_channels, 1)
        
        # 比例因子
        self.scale = self.reduced_channels ** -0.5
        
    def forward(self, x):
        # 原始特征
        identity = x
        
        # 通道降维
        x_reduced = self.conv_reduce(x)
        
        # 自注意力计算
        q = self.query(x_reduced)
        k = self.key(x_reduced)
        v = self.value(x_reduced)
        
        # 重塑张量以进行注意力计算
        b, c, h, w = q.shape
        q = q.view(b, c, -1).permute(0, 2, 1)  # [b, hw, c]
        k = k.view(b, c, -1)                   # [b, c, hw]
        v = v.view(b, c, -1).permute(0, 2, 1)  # [b, hw, c]
        
        # 计算注意力分数
        attn = torch.bmm(q, k) * self.scale     # [b, hw, hw]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(attn, v)               # [b, hw, c]
        out = out.permute(0, 2, 1).view(b, c, h, w)
        
        # 通道恢复并残差连接
        out = self.conv_expand(out)
        return out + identity

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

class DeformableAttention(nn.Module):
    """可变形卷积注意力模块
    
    使用简化版可变形卷积实现，通过学习空间偏移来适应不同形状的目标
    """
    def __init__(self, in_channels):
        super().__init__()
        # 偏移预测
        self.offset_conv = nn.Conv2d(in_channels, 18, 3, padding=1)  # 2*3*3 = 18个偏移参数
        
        # 特征提取
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 预测偏移
        offset = self.offset_conv(x)
        
        # 简化版可变形卷积实现
        b, c, h, w = x.shape
        
        # 创建基本网格
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [b, h, w, 2]
        
        # 重塑偏移并添加到网格
        offset = offset.permute(0, 2, 3, 1).reshape(b, h, w, 9, 2)  # [b, h, w, 9, 2]
        positions = []
        
        # 3x3卷积核的相对位置
        for i in range(-1, 2):
            for j in range(-1, 2):
                idx = (i+1)*3 + (j+1)
                pos = grid + offset[:, :, :, idx, :]
                
                # 归一化到[-1, 1]
                pos_x = 2.0 * pos[:, :, :, 0].clone() / max(w - 1, 1) - 1.0
                pos_y = 2.0 * pos[:, :, :, 1].clone() / max(h - 1, 1) - 1.0
                pos = torch.stack((pos_x, pos_y), dim=3)  # [b, h, w, 2]
                positions.append(pos)
        
        # 使用grid_sample实现可变形采样
        sampled_feats = []
        for pos in positions:
            sampled = F.grid_sample(x, pos, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_feats.append(sampled)
        
        # 合并所有采样特征
        deformed_feat = sum(sampled_feats) / len(sampled_feats)
        
        # 进一步处理
        out = self.act(self.bn(self.conv(deformed_feat)))
        
        return out
        
class MultiResolutionFusion(nn.Module):
    """多分辨率特征融合模块
    
    结合RD-UNet和Cell-Net的思想，提高对不同分辨率目标的检测能力
    """
    def __init__(self, in_channels):
        super().__init__()
        # 多尺度特征提取 - 使用不同膨胀率的空洞卷积
        self.dilation1 = nn.Conv2d(in_channels, in_channels//4, 3, padding=1, dilation=1)
        self.dilation2 = nn.Conv2d(in_channels, in_channels//4, 3, padding=2, dilation=2)
        self.dilation3 = nn.Conv2d(in_channels, in_channels//4, 3, padding=3, dilation=3)
        self.dilation4 = nn.Conv2d(in_channels, in_channels//4, 3, padding=4, dilation=4)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.dilation1(x)
        feat2 = self.dilation2(x)
        feat3 = self.dilation3(x)
        feat4 = self.dilation4(x)
        
        # 拼接
        multi_scale = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        
        # 融合
        fused = self.fusion(multi_scale)
        
        # 残差连接
        return x + fused

class YeastAttention(nn.Module):
    """酵母细胞检测的增强注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        # 基本注意力模块
        self.cbam = CBAM(in_channels)
        
        # 增加尺度自适应注意力
        self.scale_attention = ScaleAdaptiveAttention(in_channels)
        
        # 增加可变形注意力 - 适应不同形状的细胞
        self.deform_attention = DeformableAttention(in_channels)
        
        # 增加多分辨率融合
        self.multi_res_fusion = MultiResolutionFusion(in_channels)
        
        # 自注意力模块
        self.psa = PSA(in_channels)
        
        # 最终特征融合
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # 原始特征
        identity = x
        
        # 应用基础CBAM
        x = self.cbam(x)
        
        # 应用尺度自适应注意力
        x = self.scale_attention(x)
        
        # 应用可变形注意力 - 适应不同形状
        x = self.deform_attention(x)
        
        # 应用多分辨率融合
        x = self.multi_res_fusion(x)
        
        # 应用部分自注意力增强特征
        x = self.psa(x)
        
        # 最终处理
        x = self.act(self.bn(self.conv(x)))
        
        # 残差连接
        return x + identity 