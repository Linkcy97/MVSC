import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils.utils import save_imgs
from models.distortion import *


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 

    psnr_loss = MSE().calculate_psnr
    if config.datasets == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    loss_list, psnr_list, ms_ssim_list = [], [], []
    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        if config.datasets == 'CIFAR10':
            images, _ = data
        else:
            images = data
        images = images.cuda(non_blocking=True).float()

        out, cbr, snr = model(images)
        loss = criterion(out*255., images*255.)

        loss.backward()
        optimizer.step()
        

        loss_list.append(loss.item())
        psnr_list.append(psnr_loss(out, images).item())
        ms_ssim_list.append(1 - CalcuSSIM(images, out.clamp(0., 1.)).mean().item())        

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info =( f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, cbr: {cbr:.4f}, snr: {snr},'
                       f'psnr: {np.mean(psnr_list):.4f}, ms_ssim: {np.mean(ms_ssim_list):.4f}, lr: {now_lr}')
            print(log_info)
            logger.info(log_info)
    # scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    multiple_snr = config.multiple_snr
    loss_list, psnr_list, ms_ssim_list = [[] for _ in range(len(multiple_snr))], \
                                         [[] for _ in range(len(multiple_snr))], \
                                         [[] for _ in range(len(multiple_snr))]

    psnr_loss = MSE().calculate_psnr
    if config.datasets == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for data in tqdm(test_loader):
                if config.datasets == 'CIFAR10':
                    img, _ = data
                else:
                    img = data
                img = img.cuda(non_blocking=True).float()

                out, cbr, snr = model(img,multiple_snr[i])
                loss = criterion(out*255., img*255.)

                loss_list[i].append(loss.item())
                psnr_list[i].append(psnr_loss(out, img).item())
                ms_ssim_list[i].append(1 - CalcuSSIM(img, out.clamp(0., 1.)).mean().item())

        loss_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in loss_list]
        psnr_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in psnr_list]
        ms_ssim_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in ms_ssim_list]
        log_info = (f'val epoch: {epoch}, loss: {loss_list_avg}, cbr: {cbr}, snr: {snr},'
                    f'psnr: {psnr_list_avg}, ms_ssim: {ms_ssim_list_avg}')
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list_avg)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    multiple_snr = config.multiple_snr
    loss_list, psnr_list, ms_ssim_list = [[] for _ in range(len(multiple_snr))], \
                                         [[] for _ in range(len(multiple_snr))], \
                                         [[] for _ in range(len(multiple_snr))]

    psnr_loss = MSE().calculate_psnr
    if config.datasets == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    

    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for j, data in enumerate(tqdm(test_loader)):
                if config.datasets == 'CIFAR10':
                    img = data
                else:
                    img = data
                img  = img.cuda(non_blocking=True).float()

                out, cbr, snr = model(img,multiple_snr[i])
                loss = criterion(out*255., img*255.)

                loss_list[i].append(loss.item())
                psnr_list[i].append(psnr_loss(out, img).item())
                ms_ssim_list[i].append(1 - CalcuSSIM(img, out.clamp(0., 1.)).mean().item())

                if config.test_datasets == 'Kodak':
                    save_imgs(img, out, j, config.work_dir + 'outputs/', test_data_name='_%ssnr'%multiple_snr[i])
                else:
                    if i == j:
                        save_imgs(img, out, i, config.work_dir + 'outputs/', test_data_name='_%ssnr'%multiple_snr[i])

        loss_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in loss_list]
        psnr_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in psnr_list]
        ms_ssim_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in ms_ssim_list]
        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
        log_info = (f'test of best model, loss: {loss_list_avg} , cbr: {cbr}, snr: {snr},'
                    f' psnr: {psnr_list_avg}, ms_ssim: {ms_ssim_list_avg}')
        print(log_info)

    return loss_list_avg, psnr_list_avg, ms_ssim_list_avg