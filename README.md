# MVSC: Mamba Vision based Semantic Communication for Image Transmission with SNR Estimation

Pytorch implementation for [MVSC: Mamba Vision based Semantic Communication for Image Transmission with SNR Estimation | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/10975284)



## Introduction

This paper proposes a novel semantic communication approach named **Mamba Vision-based Semantic Communication (MVSC)** for image transmission with integrated Signal-to-Noise Ratio (SNR) estimation. Unlike prior works that assume the SNR of the received signal is known and input a predetermined SNR value into a deep learning (DL) network, MVSC introduces an implicit SNR estimation module, allowing the network to infer channel conditions for SNR adaptation. To further improve performance, we propose the MVSC4, a joint-optimized of MVSC, which is trained using a multi-task learning strategy that simultaneously optimizes image reconstruction, SNR estimation, signal denoising, and image classification. This joint optimization enhances the network’s robustness to varying SNR conditions, particularly in low-SNR environments. Comparative experiments on CIFAR-10 and Kodak datasets demonstrate that MVSC4 outperforms both CNN-based and Transformer-based methods in terms of Peak Signal-to-Noise Ratio (PSNR) and Multiscale Structural Similarity (MS-SSIM). The results demonstrate the effectiveness and robustness of the proposed approach.



## Python environment

```shell
einops==0.8.1
mamba_ssm==1.1.0
matplotlib==3.10.3
numpy==2.2.5
Pillow==11.2.1
scipy==1.15.3
sionna==0.19.0
tensorboardX==2.6.2.2
tensorboardX==2.6.2.2
tensorflow==2.14.0
thop==0.1.1.post2209072238
timm==0.4.12
torch==2.1.1+cu118
torchvision==0.16.1+cu118
tqdm==4.66.4

```

## Usage

```shell
## setting config.py and run train.py
python train.py         

## run test.py to get result. This step is generally not needed, as the test results are already generated during training.
python test.py

```

## Citation

If this work is useful for your research, please cite:

```tex
@article{li2025mvsc,
  title={MVSC: Mamba Vision based Semantic Communication for Image Transmission with SNR Estimation},
  author={Li, Chongyang and Zhang, Tianqian and Liu, Shouyin},
  journal={IEEE Communications Letters},
  year={2025},
  publisher={IEEE}
}
```

## Related links

- SwinJSCC:https://github.com/semcomm/SwinJSCC
- MambaVision:https://github.com/NVlabs/MambaVision
- Sionna for Next Generation Physical Layer research:https://github.com/NVlabs/sionna
- BPG image encoder and decoder: https://bellard.org/bpg
- CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html

Thank you for your outstanding contributions!