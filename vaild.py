# -*- coding: utf-8 -*-   
# Author       : Li Chongyang of ACTL 
# Email        : lichongyang2016@163.com
# Date         : 2024-07-11 08:14:27
# LastEditors  : Li Chongyang of ACTL
# LastEditTime : 2024-07-12 09:30:17
# FilePath     : \VM-UNet-main\test.py

import os
import torch
from models.mamba_vision import MVSC
from models.distortion import *
from classify_net import ResNet8
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
    train_loader, val_loader, test_loader = get_loader(config)

    print('#----------Prepareing Model----------#')
    sc_model = MVSC(config)
    sc_model = sc_model.cuda()

    cl_model = ResNet8({'in_channels': 3, 'out_channels': 10, 'activation': 'relu'})
    cl_model = cl_model.cuda()
    print('#----------Prepareing loss, opt, sch and amp----------#')
    psnr_crit = config.psnr_crit

    config.work_dir = 'results/' + 'CIFAR10_2024-09-03_11-00-40' + '/'
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('train', log_dir)

    if os.path.exists(os.path.join(config.work_dir, 'checkpoints/best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth')
        sc_model.load_state_dict(best_weight,strict=False)
        cl_model.load_state_dict(torch.load('classify.pth'))
        psnr = val_one_epoch(
                train_loader,
                sc_model,
                psnr_crit,
                1,
                logger,
                config
            )
            


if __name__ == '__main__':
    config = setting_config
    main(config)