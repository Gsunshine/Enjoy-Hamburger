# Enjoy-Hamburger 🍔

Official implementation of Hamburger, *[Is Attention Better Than Matrix Decomposition?](https://openreview.net/forum?id=1FvkSpWosOl)* (ICLR 2021)

Under construction.

## Update

- 2020.04.13 - Add poster and thumbnail icon for ICLR 2021.

## Introduction

This repo provides the official implementation of Hamburger for further research. We sincerely hope that this paper can bring you inspiration about the Attention Mechanism, especially how **the low-rankness and the optimization-driven method** can help model the so-called *Global Information* in deep learning.

We model the global context issue as a low-rank completion problem and show that its optimization algorithms can help design global information blocks. This paper then proposes a series of Hamburgers, in which we employ the optimization algorithms for solving MDs to factorize the input representations into sub-matrices and reconstruct a low-rank embedding. Hamburgers with different MDs can perform favorably against the popular global context module self-attention when carefully coping with gradients back-propagated through MDs.

![contents](assets/Hamburger.jpg)

We are working on some exciting topics. Please wait for our new papers!

Enjoy Hamburger, please!

## Organization

This section introduces the organization of this repo.

**We strongly recommend the readers to read the blog (incoming soon) as a supplement to the paper!**

- blog.
  - Some random thoughts about Hamburger and beyond.
  - Possible directions based on Hamburger.
  - FAQ.
- seg.
  - We provide the PyTorch implementation of Hamburger (V1) in the paper and an enhanced version (V2) flavored with Cheese. Some experimental features are included in V2+.
  - We release the codebase for systematical research on the PASCAL VOC dataset, including the two-stage training on the `trainaug` and `trainval` datasets and the MSFlip test.
  - We offer three checkpoints of HamNet, in which one is 85.90+ with the test server [link](http://host.robots.ox.ac.uk:8080/anonymous/NEHYHH.html), while the other two are 85.80+ with the test server [link 1](http://host.robots.ox.ac.uk:8080/anonymous/HEBCIV.html) and [link 2](http://host.robots.ox.ac.uk:8080/anonymous/3VNCPH.html). You can reproduce the test results using the checkpoints combined with the MSFlip test code.
  - Statistics about HamNet that might ease further research.
- gan.
  - Official implementation of Hamburger in TensorFlow.
  - Data preprocessing code for using ImageNet in tensorflow-datasets. (Possibly useful if you hope to run the JAX code of [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol) or other ImageNet training code with the Cloud TPUs.)
  - Training and evaluation protocol of HamGAN on the ImageNet.
  - Checkpoints of HamGAN-strong and HamGAN-baby.

TODO:

- [ ] README doc for HamGAN.
- [ ] PyTorch Hamburger with less encapsulation.
- [ ] Suggestions for using and further developing Hamburger.
- [ ] Blog in both English and Chinese.
- [ ] ~~We also consider adding a collection of popular context modules to this repo.~~ It depends on the time. No Guarantee. Perhaps GuGu 🕊️ (which means standing someone up).

## Citation

If you find our work interesting or helpful to your research, please consider citing Hamburger. :)

```bib
@inproceedings{
    ham,
    title={Is Attention Better Than Matrix Decomposition?},
    author={Zhengyang Geng and Meng-Hao Guo and Hongxu Chen and Xia Li and Ke Wei and Zhouchen Lin},
    booktitle={International Conference on Learning Representations},
    year={2021},
}
```

## Contact

Feel free to contact me if you have additional questions or have interests in collaboration. Please drop me an email at zhengyanggeng@gmail.com. Find me at [Twitter](https://twitter.com/ZhengyangGeng). Thank you!

Response to recent emails may be slightly delayed to March 26th due to the deadlines of ICLR. I feel sorry, but people are always deadline-driven. QAQ

## Acknowledgments

Our research is supported with Cloud TPUs from Google's [Tensorflow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc). Nice and joyful experience with the TFRC program. Thank you!

We would like to sincerely thank [EMANet](https://github.com/XiaLiPKU/EMANet), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [YLG](https://github.com/giannisdaras/ylg/tree/train), and [TF-GAN](https://github.com/tensorflow/gan) for their awesome released code.
