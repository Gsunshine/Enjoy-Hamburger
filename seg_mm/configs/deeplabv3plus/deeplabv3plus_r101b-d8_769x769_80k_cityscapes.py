_base_ = './deeplabv3plus_r50-d8_769x769_80k_cityscapes.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
