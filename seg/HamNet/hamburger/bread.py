# -*- coding: utf-8 -*-
"""
Hamburger for Pytorch

@author: Gsunshine
"""

from functools import partial

import numpy as np
import settings
import torch
from sync_bn.nn.modules import SynchronizedBatchNorm2d
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)


class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size):
        if kernel_size == 1:
            return 0
        elif kernel_size == 3:
            return 1

    def __init__(self, in_c, out_c,
                 kernel_size=1, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel_size)

        self.conv = nn.Conv2d(in_c, out_c,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups,
                              bias=False)
        self.bn = norm_layer(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x

