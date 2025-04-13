"""
高分辨率模块

提供高分辨率特征提取和多尺度特征融合功能。
基于HRNet和HRNetV2的设计理念，但进行了轻量化和针对细胞检测的优化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class HighResolutionModule(nn.Module):
    """高分辨率特征提取模块"""
    
    def __init__(self, 
                 channels: List[int],
                 num_branches: int = 3,
                 multi_scale_output: bool = True):
        """
        初始化高分辨率模块
        
        Args:
            channels: 各分支的通道数列表
            num_branches: 分支数量
            multi_scale_output: 是否输出多尺度特征
        """
        super().__init__()
        
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.channels = channels
        
        # 创建分支
        self.branches = self._make_branches()
        
        # 创建融合层
        self.fuse_layers = self._make_fuse_layers()
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branches(self) -> nn.ModuleList:
        """创建各分支的基本块"""
        branches = nn.ModuleList()
        
        for i in range(self.num_branches):
            branches.append(self._make_branch_block(self.channels[i]))
            
        return branches
    
    def _make_branch_block(self, channels: int) -> nn.Sequential:
        """创建分支基本块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def _make_fuse_layers(self) -> nn.ModuleList:
        """创建特征融合层"""
        if self.num_branches == 1:
            return nn.ModuleList()
            
        num_branches = self.num_branches
        num_output_branches = num_branches if self.multi_scale_output else 1
        channels = self.channels
        
        fuse_layers = nn.ModuleList()
        for i in range(num_output_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:  # 低分辨率到高分辨率，需要上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:  # 相同分辨率，不需要处理
                    fuse_layer.append(nn.Identity())
                else:  # 高分辨率到低分辨率，需要下采样
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(channels[j], channels[j], kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(channels[j]),
                            nn.ReLU(inplace=True)
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(channels[i])
                    ))
                    fuse_layer.append(nn.Sequential(*ops))
            fuse_layers.append(fuse_layer)
            
        return fuse_layers
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播"""
        # 处理各分支
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        # 如果只有一个分支，直接返回
        if self.num_branches == 1:
            return [x[0]]
        
        # 融合各分支特征
        out = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                y = y + fuse_layer[j](x[j])
            out.append(self.relu(y))
            
        return out


class CellHighResolutionNet(nn.Module):
    """细胞检测专用高分辨率网络"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 base_channels: int = 32,
                 num_modules: int = 2,
                 num_branches: int = 3):
        """
        初始化细胞检测专用高分辨率网络
        
        Args:
            input_channels: 输入通道数
            base_channels: 基础通道数
            num_modules: 高分辨率模块数量
            num_branches: 分支数量
        """
        super().__init__()
        
        self.num_branches = num_branches
        
        # 初始化各分支的通道数
        self.channels = [base_channels * (2**i) for i in range(num_branches)]
        
        # 初始化层
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 创建第一阶段
        self.stage1 = self._make_stage1()
        
        # 创建第二阶段
        self.stage2 = self._make_stage2(num_modules)
        
        # 创建最终融合层
        self.final_fusion = self._make_final_fusion()
    
    def _make_stage1(self) -> nn.Sequential:
        """创建第一阶段"""
        return nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True)
        )
    
    def _make_stage2(self, num_modules: int) -> nn.ModuleList:
        """创建第二阶段的高分辨率模块"""
        modules = nn.ModuleList()
        
        # 创建分支转换层
        transition = self._make_transition()
        modules.append(transition)
        
        # 创建高分辨率模块
        for i in range(num_modules):
            # 最后一个模块输出多尺度特征，其他模块也输出多尺度特征
            modules.append(HighResolutionModule(
                channels=self.channels,
                num_branches=self.num_branches,
                multi_scale_output=True
            ))
            
        return modules
    
    def _make_transition(self) -> nn.ModuleList:
        """创建分支转换层"""
        transition = nn.ModuleList()
        
        # 第一个分支保持原样
        transition.append(nn.Identity())
        
        # 创建其他分支
        for i in range(1, self.num_branches):
            transition.append(nn.Sequential(
                nn.Conv2d(self.channels[i-1], self.channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.channels[i]),
                nn.ReLU(inplace=True)
            ))
            
        return transition
    
    def _make_final_fusion(self) -> nn.ModuleList:
        """创建最终特征融合层"""
        # 将所有分支特征上采样到最高分辨率并融合
        fusion_layers = nn.ModuleList()
        
        for i in range(1, self.num_branches):
            fusion_layers.append(nn.Sequential(
                nn.Conv2d(self.channels[i], self.channels[0], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.channels[0]),
                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
            ))
            
        return fusion_layers
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 初始化层
        x = self.stem(x)
        
        # 第一阶段
        x = self.stage1(x)
        
        # 初始化各分支
        x_list = [x]
        transition = self.stage2[0]
        
        for i in range(1, self.num_branches):
            x_list.append(transition[i](x_list[i-1]))
        
        # 高分辨率模块
        for i in range(1, len(self.stage2)):
            x_list = self.stage2[i](x_list)
        
        # 最终特征融合
        y = x_list[0]
        for i in range(1, self.num_branches):
            y = y + self.final_fusion[i-1](x_list[i])
        
        # 返回多尺度特征和融合特征
        return {
            'high_res': y,  # 高分辨率特征
            'multi_scale': x_list  # 多尺度特征
        }
