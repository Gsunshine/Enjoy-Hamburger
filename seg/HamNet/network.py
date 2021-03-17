import math
import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import settings
from hamburger import ConvBNReLU, get_hamburger
from sync_bn.nn.modules import SynchronizedBatchNorm2d

norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation,
                               bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, stride=8):
        self.inplanes = 128
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(
                    block, 512, layers[3], stride=1, dilation=2, grids=[1,2,4])
        elif stride == 8:
            self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(
                    block, 512, layers[3], stride=1, dilation=4, grids=[1,2,4])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    grids=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        if grids is None:
            grids = [1] * blocks

        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample,
                                previous_dilation=dilation))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample,
                                previous_dilation=dilation))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation*grids[i],
                                previous_dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(n_layers, stride):
    layers = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[n_layers]
    pretrained_path = {
        50:  osp.join(settings.MODEL_DIR, 'resnet50-ebb6acbb.pth'),
        101: osp.join(settings.MODEL_DIR, 'resnet101-2a57e44d.pth'),
        152: osp.join(settings.MODEL_DIR, 'resnet152-0d43d698.pth'),
    }[n_layers]

    net = ResNet(Bottleneck, layers=layers, stride=stride)
    state_dict = torch.load(pretrained_path)
    net.load_state_dict(state_dict, strict=False)

    return net


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)


class HamNet(nn.Module):
    def __init__(self, n_classes, n_layers):
        super().__init__()
        backbone = resnet(n_layers, settings.STRIDE)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4)

        C = settings.CHANNELS

        self.squeeze = ConvBNReLU(2048, C, 3)

        Hamburger = get_hamburger(settings.VERSION)
        self.hamburger = Hamburger(C, settings)
        
        self.align = ConvBNReLU(C, 256, 3)
        self.fc = nn.Sequential(nn.Dropout2d(p=0.1),
                                nn.Conv2d(256, n_classes, 1))

        # Put the criterion inside the model to make GPU load balanced
        self.crit = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL,
                                       reduction='none')

    def forward(self, img, lbl=None, size=None):
        x = self.backbone(img)

        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        x = self.fc(x)

        if size is None:
            size = img.size()[-2:]

        pred = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        if self.training and lbl is not None:
            loss = self.crit(pred, lbl)
            return loss
        else:
            return pred


def test_net():
    model = HamNet(n_classes=21, n_layers=50)
    model.eval()
    print(list(model.named_children()))
    image = torch.randn(1, 3, 513, 513)
    label = torch.zeros(1, 513, 513).long()
    pred = model(image, label)
    print(pred.size())


if __name__ == '__main__':
    test_net()
