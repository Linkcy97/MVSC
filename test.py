# -*- coding: utf-8 -*-   
# Author       : Li Chongyang of ACTL 
# Email        : lichongyang2016@163.com
# Date         : 2024-07-11 08:14:27
# LastEditors  : Li Chongyang of ACTL
# LastEditTime : 2024-07-12 09:30:17
# FilePath     : \VM-UNet-main\test.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.mamba_vision import MVSC
from models.distortion import *
import argparse
from engine import *
import os
from utils.datasets import *
from utils.utils import *
from configs.config import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):
    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()


    print('#----------Preparing dataset----------#')
    _, _, test_loader = get_loader(config)

    print('#----------Prepareing Model----------#')
    model = MVSC(config)
    model = model.cuda()


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion

    config.work_dir = 'results/' + 'CIFAR10_50mb_1-13' + '/'

    if os.path.exists(os.path.join(config.work_dir, 'checkpoints/best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth')
        model.load_state_dict(best_weight,strict=False)
        loss, psnr, msssim = test_one_epoch(
                test_loader,
                model,
                criterion,
                config,
                )
            


if __name__ == '__main__':
    config = setting_config
    main(config)