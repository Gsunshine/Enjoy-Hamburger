_base_ = './fcn_hr18_512x512_80k_loveda.py'
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w48'),
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
