# -*- coding: utf-8 -*-   
# Author       : Li Chongyang of ACTL 
# Email        : lichongyang2016@163.com
# Date         : 2024-07-11 08:14:27
# LastEditors  : Chongyang Li
# LastEditTime : 2025-05-15 10:20:32
# FilePath     : /MVSC/test.py

import os
import torch
from models.mamba_vision import MVSC
from models.djscc import Djscc
from models.distortion import *
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

    config.work_dir = 'results/' + 'MVSC_0.06_4loss_1_12_AWGN' + '/'
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('train', log_dir)

    if os.path.exists(os.path.join(config.work_dir, 'checkpoints/best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth')
        sc_model.load_state_dict(best_weight, strict=False)
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