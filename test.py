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
    _, _, test_loader, kodak_loader = get_loader(config)

    print('#----------Prepareing Model----------#')
    sc_model = MVSC(config)
    sc_model = sc_model.cuda()

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.psnr_crit

    config.work_dir = 'results/' + 'CIFAR10_2024-09-18_20-30-45' + '/'
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('train', log_dir)

    if os.path.exists(os.path.join(config.work_dir, 'checkpoints/best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth')
        sc_model.load_state_dict(best_weight,strict=False)
        score = test_one_epoch(
                test_loader,
                kodak_loader,
                sc_model,
                criterion,
                config,
                logger)
            


if __name__ == '__main__':
    config = setting_config
    main(config)