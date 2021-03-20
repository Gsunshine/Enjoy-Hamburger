# HamGAN

Official codebase for HamGAN on ImageNet.

Under construction.

## Dataset

We divide the data preprocessing into s single section because using ImageNet in tensorflow-datasets (`tfds`) can be really difficult. Since ImageNet changed the protocol to access it, new users need to apply for permission to download it, bringing additional problems for using ImageNet in `tfds`. We provide several possible solutions for different users. It is possibly useful if you hope to run the JAX code of [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol) or other ImageNet training code with the Cloud TPUs.

To use ImageNet in `tfds`, one needs to prepare `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`.

### How to get the `tar` files

If you do not have the `tar` files, there are several possible ways.

- Resister in [ImageNet](http://image-net.org/request) and apply for access permission to download it. (My application submitted one year ago have not been approved yet.)
- If your lab server has the downloaded dataset but unzipped for PyTorch Dataset,
  - set `target_dir` and `imagenet_dir` in `tar_files.sh` to the directory where you hope to save the final `tar` files and the directory where your local ImageNet are stored, respectively,
  - use `bash tar_files.sh` to `tar` the 1000 sub-class `train` sets and the `val` set to `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`.
- Or you can download the 1000 sub-class `train` files and `ILSVRC2012_img_val.tar` from this [link](https://pan.baidu.com/s/1Yg0suw4_3warFOlUZCJloA) with the password *4gdh*. `tar` the `train` sets by yourself.
  
  *The shared ImageNet dataset is allowed for non-commercial research and/or educational purposes only.*

Additionally, if you want to use VMs in Google Cloud Computing (GCP),

- create a VM with a 300G+ disk,
- download ImageNet from Baidu Netdisk via [bypy](https://github.com/houtianze/bypy) (recommended) or from your local machine via `scp`.

### How to prepare ImageNet in tensorflow-datasets

If you have the `tar` files,

- set `cache_dir` to the directory where the `tar` files are saved and `data_dir` to the directory where you want to store the processed `tf_record` in the `tfds.py`. 
- run the script.

    ```sh
    python3 tfds.py
    ```

Additionally, if you are using VMs in Google Cloud Computing (GCP),

- create a cloud storage to share the processed datasets across different VMs,
- set `STORAGE_BUCKET` in the `tfds.py` to your cloud storage name.

## Usage

## Checkpoints

HamGAN-strong

[Google Drive](https://drive.google.com/drive/folders/1de7wmmiirSNvnRLl6HNoZEy7jUV-Eht4?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/18pl0WRY0iSteGTg-DTlXZg) with password *s4i6*.

HamGAN-baby

[Google Drive](https://drive.google.com/drive/folders/1Q43BdftfrniWXx_UOVhK0toAX--ybqwn?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1Xbirn5K85rZpoAo8DXhbAA) with password *0p99*.

## Statistics

## Acknowledgments

Our research is supported with Cloud TPUs from Google's [Tensorflow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc). Nice and joyful experience with the TFRC program. Thank you!

We would like to sincerely thank [YLG](https://github.com/giannisdaras/ylg/tree/train) and [TF-GAN](https://github.com/tensorflow/gan) for their awesome released code.
