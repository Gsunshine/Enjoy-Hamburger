# BiSeNetV1

[BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/ycszen/TorchSeg/tree/master/model/bisenet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/backbones/bisenetv1.py#L266">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Semantic segmentation requires both rich spatial information and sizeable receptive field. However, modern approaches usually compromise spatial resolution to achieve real-time inference speed, which leads to poor performance. In this paper, we address this dilemma with a novel Bilateral Segmentation Network (BiSeNet). We first design a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features. Meanwhile, a Context Path with a fast downsampling strategy is employed to obtain sufficient receptive field. On top of the two paths, we introduce a new Feature Fusion Module to combine features efficiently. The proposed architecture makes a right balance between the speed and segmentation performance on Cityscapes, CamVid, and COCO-Stuff datasets. Specifically, for a 2048x1024 input, we achieve 68.4% Mean IOU on the Cityscapes test dataset with speed of 105 FPS on one NVIDIA Titan XP card, which is significantly faster than the existing methods with comparable performance.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142898839-a0a78148-848a-41b2-8682-b1f61ac004ba.png" width="70%"/>
</div>

## Citation

```bibtex
@inproceedings{yu2018bisenet,
  title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={325--341},
  year={2018}
}
```

## Results and models

### Cityscapes

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiSeNetV1 (No Pretrain) | R-18-D32 | 1024x1024  | 160000 | 5.69 | 31.77 | 74.44 | 77.05 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes_20210922_172239-c55e78e2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes_20210922_172239.log.json) |
| BiSeNetV1| R-18-D32 | 1024x1024  | 160000 | 5.69 | 31.77 | 74.37 | 76.91 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210905_220251-8ba80eff.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210905_220251.log.json) |
| BiSeNetV1 (4x8) | R-18-D32 | 1024x1024  | 160000 | 11.17 | 31.77 | 75.16 | 77.24 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes_20210905_220322-bb8db75f.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes_20210905_220322.log.json) |
| BiSeNetV1 (No Pretrain) | R-50-D32 | 1024x1024  | 160000 | 15.39 | 7.71 | 76.92 | 78.87 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes_20210923_222639-7b28a2a6.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes_20210923_222639.log.json) |
| BiSeNetV1 | R-50-D32 | 1024x1024  | 160000 | 15.39 | 7.71 | 77.68 | 79.57 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628-8b304447.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628.log.json) |

### COCO-Stuff 164k

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiSeNetV1 (No Pretrain) | R-18-D32 | 512x512 | 160000 | - | - | 25.45 | 26.15 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211022_054328-046aa2f2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211022_054328.log.json) |
| BiSeNetV1| R-18-D32 | 512x512  | 160000 | 6.33 | 74.24 | 28.55 | 29.26 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211023_013100-f700dbf7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211023_013100.log.json) |
| BiSeNetV1 (No Pretrain) | R-50-D32 | 512x512  | 160000 | - | - | 29.82 | 30.33 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r50-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r50-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211101_040616-d2bb0df4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r50-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211101_040616.log.json) |
| BiSeNetV1 | R-50-D32 | 512x512  | 160000 | 9.28 | 32.60 | 34.88 | 35.37 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r50-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211101_181932-66747911.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r50-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211101_181932.log.json) |
| BiSeNetV1(No Pretrain) | R-101-D32 | 512x512  | 160000 | - | - | 31.14 | 31.76 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r101-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r101-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r101-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211102_164147-c6b32c3b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r101-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r101-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211102_164147.log.json) |
| BiSeNetV1 | R-101-D32 | 512x512  | 160000 | 10.36 | 25.25 | 37.38 | 37.99 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r101-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r101-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r101-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211101_225220-28c8f092.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r101-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r101-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211101_225220.log.json) |

Note:

- `4x8`: Using 4 GPUs with 8 samples per GPU in training.
- For BiSeNetV1 on Cityscapes dataset, default setting is 4 GPUs with 4 samples per GPU in training.
- `No Pretrain` means the model is trained from scratch.
