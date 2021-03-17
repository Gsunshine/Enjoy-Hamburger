# HamNet

Official codebase for HamNet on the PASCAL VOC dataset.

## Introduction

---
We provide the PyTorch implementation of Hamburger (V1) in the paper and an enhanced version (V2) flavored with additional Cheese (1*1 Conv -> BN -> ReLU) between the Ham and the Upper Bread. Some experimental features like the multi-head operation and the spatial-channel ensemble are supported by V2+. This repo contains all three types of Ham, namely NMF, CD, and VQ.

![contents](../assets/Hamburger.jpg)

The codebase is modified from [EMANet](https://github.com/XiaLiPKU/EMANet) to support systematical research on the PASCAL VOC dataset, including the two-stage training on the `trainaug` and `trainval` datasets and the MSFlip test. The output of this codebase can be directly submitted to the test server.

From the bottom of our hearts, we hope that our work can inspire and ease your further research.

## Usage

---

### 1. Installation

- Download this repo.

    ```sh
    git clone https://github.com/Gsunshine/Enjoy-Hamburger.git
    ```

- Install the dependencies.

    ```sh
    pip install -r seg/requirements.txt
    ```

### 2. Path

- Set the environmental variables in the `data_settings.py`, including `DATA_ROOT` for the directory to the PASCAL VOC dataset, `MODEL_DIR` to save the pretrained backbone and later trained models, `TEST_SAVE_DIR` to save the MSFlip test results.

### 3. Dataset

- Download the [PASCAL VOC2012](http://host.robots.ox.ac.uk:8080/pascal/VOC/) dataset and the the [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) dataset.
- Prepare the trainaug set according to the [English blog 1](https://github.com/Media-Smart/vedaseg), [English blog 2](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/) or the [Chinese blog](https://blog.csdn.net/kxh123456/article/details/103972943).
- Name the label directory as `SegmentationClassAug`.
- Move the dataset to the `DATA_ROOT` in the `data_settings.py`.

Or you can download the processed datasets from this [link](https://pan.baidu.com/s/1jhfNkU-_O9v-n4SFK4C0zA) with password *yuyz*.

### 4. Pretrained Backbone

- Download the ImageNet pretrained ResNet from [Baidu Netdisk](https://pan.baidu.com/s/1sPht7Qiy7Hv9M5uYVT6beg). The password is *lcfu*. Or you can download the backbone models from [Google Drive](https://drive.google.com/drive/folders/1FXwv1j-qYu2x8dLCvTVWJjAuWyp_Gq_I?usp=sharing).
- Move the models to the `MODEL_DIR` in the `data_settings.py`.

### 5. Training

- To train a model towards test submission, run the script to start the two-stage training on the `trainaug` and `trainval` datasets. The code runs for about 3 days with at least 8 * 11G-memory GPUs.

    ```sh
    sh train.sh
    ```

  - The codebase will output two checkpoints `best_trainaug.pth` and `final_trainaug.pth` trained on the `trainaug` set, and two checkpoints `best.pth` and `final.pth` further trained on the `trainval` set to `MODEL_DIR/EXP_NAME`.
  - The code will test the model on the PASCAL VOC validation set every `ITER_VAL` iterations and record the best results.
  - The single scale test results of `best_trainaug.pth` and `final_trainaug.pth` are recorded in the `eval_log/EXP_NAME`.

- To train a models for the validation set, change `RUN_FOR_TEST` to `False`.
- The default settings lead to HamNet with Hamburger V2, which is the same as this [checkpoint](http://host.robots.ox.ac.uk:8080/anonymous/NEHYHH.html). To enable Hamburger V2+, swap `settings_V2+` and `settings.py`.
- Set `VERSION` to `V1`, `V2`, or `V2+`, `HAM_TYPE` to `NMF`, `CD`, or `VQ` in the `settings.py`. Recommended hyper-parameters *for the PASCAL VOC dataset* have been shown in the `settings.py`.
- To use `CD` or `VQ` Ham, change `INV_T` to 10 or 100. Disable `RAND_INIT` for `CD`. Plus, you need to manually employ the `online_update` method.

### 6. Test

- To run the MSFlip test, use the script.

    ```sh
    sh msflip_test.sh
    ```

- Move the test results from `/TEST_SAVE_DIR/EXP_NAME` to `results/VOC2012/Segmentation/comp6_test_cls`.
- Submit the test results to the PASCAL VOC test server in a `tar` file.

## Checkpoints

---
We offer three checkpoints of HamNet, in which one is 85.90+ with the test server [link](http://host.robots.ox.ac.uk:8080/anonymous/NEHYHH.html), while the other two are 85.80+ with the test server [link 1](http://host.robots.ox.ac.uk:8080/anonymous/HEBCIV.html) and [link 2](http://host.robots.ox.ac.uk:8080/anonymous/3VNCPH.html), respectively.

| Num | Version | mIoU |  r  | Google Drive | Baidu Netdisk | Password |
| :-- | :------ | :--: | :-: | :----------: | :-----------: | :------: |
|  1  | V2      | 85.94 | 64 | [link](https://drive.google.com/drive/folders/1Rz-5TZ46YIYEgZ-NIt-E3I9LUvIkbF2-?usp=sharing) | [link](https://pan.baidu.com/s/150mHqnQZ-t_J1wkfV1R6EQ) | 4hmv |
|  2  | V2+     | 85.82 | 512 | [link](https://drive.google.com/drive/folders/1wjJRHkCg3chuXoGLomFEvoCd-ZXD4yLm?usp=sharing) | [link](https://pan.baidu.com/s/1VGhiwNpmWbLzUrqGg_K-AQ) | o8dn |
|  3  | V2+     | 85.81 | 512 | [link](https://drive.google.com/drive/folders/1GAjTy_M7VfGnRv6t9s83L3ez2hpdxF2V?usp=sharing) | [link](https://pan.baidu.com/s/13wck9IEyjwXEGHFqNDMvUw) | 7ftr |

You can reproduce the test results using the checkpoints combined with the MSFlip test code.

- Download the checkpoints to a directory under `MODEL_DIR` in the `data_settings.py`. 
- Set the `EXP_NAME` in the `settings.py` to the directory.
- Run `msflip_test.sh`.

To test HamNet with Hamburger V2+, use `settings_V2+.py`.

Note that the V2+ checkpoints are also simple and *do not* enable the multi-head operation or the spatial-channel ensemble. You can find their settings in the `settings_V2+.py`.

## Statistics

---

## Acknowledgments

---
We would like to sincerely thank [EMANet](https://github.com/XiaLiPKU/EMANet) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) for their awesome released code.
