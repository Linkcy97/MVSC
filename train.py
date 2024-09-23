import sys
import os
# 获取主程序文件的目录
main_dir = os.path.dirname(os.path.abspath(__file__))
# 将主程序目录添加到 sys.path
if main_dir not in sys.path:
    sys.path.append(main_dir)
import torch
from classify_net import ResNet8
from utils.datasets import get_loader
from tensorboardX import SummaryWriter
from models.mamba_vision import MVSC
from models.djscc import Djscc
from engine import *
import os
import shutil
import sys
from utils.utils import *
from configs.config import setting_config

import warnings
warnings.filterwarnings("ignore")

def delete_empty_checkpoints(root_dir):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name == 'checkpoints':
                # 检查 checkpoints 文件夹是否为空
                if not os.listdir(dir_path):
                    # 获取包含 checkpoints 文件夹的父文件夹路径
                    parent_dir = os.path.dirname(dir_path)
                    print(f"Deleting folder: {parent_dir}")
                    shutil.rmtree(parent_dir)


def main(config):
    delete_empty_checkpoints('results')
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)
    # copy model file to work_dir
    shutil.copy('models/mamba_vision.py', config.work_dir)
    shutil.copy('models/djscc.py', config.work_dir)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()


    print('#----------Preparing dataset----------#')
    train_loader, val_loader, test_loader, kodak_loader = get_loader(config)


    print('#----------Prepareing Model----------#')
    model = MVSC(config)
    model = model.cuda()
    cl_model = ResNet8({'in_channels': 3, 'out_channels': 10, 'activation': 'relu'})
    cl_model = cl_model.cuda()
    cl_model.load_state_dict(torch.load('classify.pth'))
    cal_params_flops(model, config.input_size_h, logger)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    psnr_crit = config.psnr_crit
    snr_crit = config.snr_crit
    cla_crit = config.cla_crit
    signal_crit = config.signal_crit
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Set other params----------#')
    max_score = 0
    start_epoch = 1
    min_epoch = 1



    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            optimizer,
            psnr_crit,
            snr_crit,
            cla_crit,
            signal_crit,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        score = val_one_epoch(
                val_loader,
                model,
                psnr_crit,
                snr_crit,
                signal_crit,
                epoch,
                logger,
                config
            )

        if score > max_score:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            print('----------Best Model Saved----------')
            logger.info('----------Best Model Saved----------')
            max_score = score
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'max_score': max_score,
                'min_epoch': min_epoch,
                'score': score,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(  test_loader,
                                kodak_loader,
                                model,
                                psnr_crit,
                                config,
                                logger)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-max_score{max_score:.4f}.pth')
        )      
        logger.info(f'best-epoch{min_epoch}-max_score{max_score:.4f}.pth')


if __name__ == '__main__':
    config = setting_config
    main(config)