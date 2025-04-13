#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
颈部网络模块

提供特征金字塔网络(FPN)和特征融合功能，连接骨干网络和检测头。
"""

from .fpn import LightweightFeaturePyramid

__all__ = [
    'LightweightFeaturePyramid'
] 