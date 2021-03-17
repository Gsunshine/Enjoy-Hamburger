# -*- coding: utf-8 -*-
"""
Hamburger for Pytorch

@author: Gsunshine
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .bread import ConvBNReLU, norm_layer
from .ham import get_hams


class HamburgerV1(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()

        ham_type = getattr(args, 'HAM_TYPE', 'NMF')

        D = getattr(args, 'MD_D', 512)

        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv2d(in_c, D, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv2d(in_c, D, 1)

        HAM = get_hams(ham_type)
        self.ham = HAM(args)
        
        self.upper_bread = nn.Sequential(nn.Conv2d(D, in_c, 1, bias=False),
                                         norm_layer(in_c))
        
        self.shortcut = nn.Sequential()
        
        self._init_weight()
        
        print('ham', HAM)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.upper_bread(x)

        x = F.relu(x + shortcut, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


class HamburgerV2(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()

        ham_type = getattr(args, 'HAM_TYPE', 'NMF')

        C = getattr(args, 'MD_D', 512)

        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv2d(in_c, C, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv2d(in_c, C, 1)

        HAM = get_hams(ham_type)
        self.ham = HAM(args)

        self.cheese = ConvBNReLU(C, C)
        self.upper_bread = nn.Conv2d(C, in_c, 1, bias=False)

        self.shortcut = nn.Sequential()

        self._init_weight()

        print('ham', HAM)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.cheese(x)
        x = self.upper_bread(x)

        x = F.relu(x + shortcut, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


class HamburgerV2Plus(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()

        ham_type = getattr(args, 'HAM_TYPE', 'NMF')

        S = getattr(args, 'MD_S', 1)
        D = getattr(args, 'MD_D', 512)
        C = S * D

        self.dual = getattr(args, 'DUAL', True)
        if self.dual:
            C = 2 * C

        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv2d(in_c, C, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv2d(in_c, C, 1)

        HAM = get_hams(ham_type)
        if self.dual:
            args.SPATIAL = True
            self.ham_1 = HAM(args)
            args.SPATIAL = False
            self.ham_2 = HAM(args)
        else:
            self.ham = HAM(args)

        factor = getattr(args, 'CHEESE_FACTOR', S)
        if self.dual:
            factor = 2 * factor

        self.cheese = ConvBNReLU(C, C // factor)
        self.upper_bread = nn.Conv2d(C // factor, in_c, 1, bias=False)

        zero_ham = getattr(args, 'ZERO_HAM', True)
        if zero_ham:
            coef_ham_init = 0.
        else:
            coef_ham_init = 1.

        self.coef_shortcut = nn.Parameter(torch.tensor([1.]))
        self.coef_ham = nn.Parameter(torch.tensor([coef_ham_init]))

        self.shortcut = nn.Sequential()

        self._init_weight()

        print('ham', HAM)
        print('dual', self.dual)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.lower_bread(x)

        if self.dual:
            x = x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
            x_1 = self.ham_1(x.narrow(1, 0, 1).squeeze(dim=1))
            x_2 = self.ham_2(x.narrow(1, 1, 1).squeeze(dim=1))
            x = torch.cat([x_1, x_2], dim=1)
        else:
            x = self.ham(x)
        x = self.cheese(x)
        x = self.upper_bread(x)
    
        x = self.coef_ham * x + self.coef_shortcut * shortcut
        x = F.relu(x, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


def get_hamburger(version):
    burgers = {'V1':HamburgerV1,
               'V2':HamburgerV2,
               'V2+': HamburgerV2Plus}

    assert version in burgers

    return burgers[version]


