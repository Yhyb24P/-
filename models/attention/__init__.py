#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
注意力机制模块

提供各种注意力机制实现，如通道注意力、空间注意力和CBAM等。
"""

from .cbam import LightweightCBAM, LightweightChannelAttention, LightweightSpatialAttention
from .modules import YeastAttention

__all__ = [
    'LightweightCBAM',
    'LightweightChannelAttention', 
    'LightweightSpatialAttention',
    'YeastAttention'
] 