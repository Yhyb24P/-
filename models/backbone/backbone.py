import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels//2, 1, 1, 0)
        self.conv2 = ConvBlock(channels//2, channels, 3, 1, 1)
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class SCDown(nn.Module):
    """空间通道解耦降采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 空间降维 - 深度卷积
        self.spatial_down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )
        # 通道转换 - 点卷积
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        x = self.spatial_down(x)
        x = self.channel_conv(x)
        return x

class LargeKernelConv(nn.Module):
    """大核卷积模块 - 提高小目标检测能力"""
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        # 使用深度可分离大核卷积
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(0.1)
        
        # 点卷积调整通道
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.dw_conv(x)))
        x = self.act2(self.bn2(self.pw_conv(x)))
        return x

class CSPBlock(nn.Module):
    """CSP (Cross Stage Partial) 块"""
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels//2)
        self.conv2 = ConvBlock(in_channels, out_channels//2)
        self.blocks = nn.Sequential(*[ResBlock(out_channels//2) for _ in range(num_blocks)])
        self.conv3 = ConvBlock(out_channels, out_channels)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.blocks(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class HighResolutionModule(nn.Module):
    """高分辨率处理模块
    
    针对高分辨率图像进行优化的特征提取模块，
    参考TasselNetV2+和Cell-Net方法
    """
    def __init__(self, channels):
        super().__init__()
        # 分支1：普通卷积
        self.branch1 = ConvBlock(channels, channels//4)
        
        # 分支2：空洞卷积 - 增大感受野
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, 1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1)
        )
        
        # 分支3：深度可分离卷积 - 降低计算量
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels//4, 1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1)
        )
        
        # 分支4：全局上下文
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1)
        )
        
        # 融合层
        self.fusion = ConvBlock(channels, channels)
        
    def forward(self, x):
        # 特征大小
        h, w = x.shape[2:]
        
        # 计算分支
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x4 = F.interpolate(x4, size=(h, w), mode='nearest')
        
        # 融合特征
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fusion(x)
        
        return x

class EfficientDownsample(nn.Module):
    """高效下采样模块
    
    减少计算量的同时保持特征质量
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 轻量化下采样 - 两个步骤分离
        self.pool = nn.AvgPool2d(2, 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class YeastBackbone(nn.Module):
    """增强版YOLOv10主干网络"""
    def __init__(self):
        super().__init__()
        # 初始卷积层 - 使用大核卷积增强对小目标的感知能力
        self.stem = nn.Sequential(
            ConvBlock(3, 32, 3, 1),
            LargeKernelConv(32, 64, 7)
        )
        
        # 高分辨率模块
        self.hires_module = HighResolutionModule(64)
        
        # 特征提取阶段
        self.stage1 = CSPBlock(64, 64, 1)
        self.stage2 = CSPBlock(128, 128, 2)
        self.stage3 = CSPBlock(256, 256, 3)
        self.stage4 = CSPBlock(512, 512, 4)
        self.stage5 = CSPBlock(512, 1024, 5)
        
        # 高效下采样 - 替代原有的SCDown，提高计算效率
        self.down1 = nn.ModuleDict({
            'standard': SCDown(64, 64),
            'efficient': EfficientDownsample(64, 64)
        })
        self.down2 = nn.ModuleDict({
            'standard': SCDown(64, 128),
            'efficient': EfficientDownsample(64, 128)
        })
        self.down3 = nn.ModuleDict({
            'standard': SCDown(128, 256),
            'efficient': EfficientDownsample(128, 256)
        })
        self.down4 = nn.ModuleDict({
            'standard': SCDown(256, 512),
            'efficient': EfficientDownsample(256, 512)
        })
        
        # 创建轻量级跨层连接
        self.p5_to_p3_reduce = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # 局部注意力模块 - 增强小目标特征
        self.local_attention = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, groups=4),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 判断是否是高分辨率图像
        is_high_res = max(x.shape[2:]) > 800
        
        # 初始特征提取
        x = self.stem(x)
        
        # 对高分辨率图像应用专门的处理
        if is_high_res:
            x = self.hires_module(x)
            
            # 使用高效下采样路径
            x1 = self.stage1(self.down1['efficient'](x))    # 1/4
            x2 = self.stage2(self.down2['efficient'](x1))   # 1/8
            x3 = self.stage3(self.down3['efficient'](x2))   # 1/16
            x4 = self.stage4(self.down4['efficient'](x3))   # 1/32
            x5 = self.stage5(x4)                           # 1/32
            
            # 增强小目标特征 - 从高层向低层传递语义信息
            p5_reduce = self.p5_to_p3_reduce(x5)
            p5_up = F.interpolate(p5_reduce, size=x3.shape[2:], mode='nearest')
            
            # 应用局部注意力
            attn = self.local_attention(x3)
            x3 = x3 * attn + p5_up
        else:
            # 标准特征提取路径
            x1 = self.stage1(self.down1['standard'](x))    # 1/4
            x2 = self.stage2(self.down2['standard'](x1))   # 1/8
            x3 = self.stage3(self.down3['standard'](x2))   # 1/16
            x4 = self.stage4(self.down4['standard'](x3))   # 1/32
            x5 = self.stage5(x4)                           # 1/32
        
        return x3, x4, x5  # 返回三个尺度的特征图
