import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from utils.utils import save_imgs, AverageMeter
from models.distortion import *
import time
from models.AutomaticWeightedLoss import AutomaticWeightedLoss
    

d_snr = DiscreteSNRLayer([-7., -4., 0., 4., 7.])

def train_one_epoch(train_loader,
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
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 

    psnr_loss_cal = MSE().calculate_psnr
    if config.datasets == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    losses, psnrs, ms_ssims, snrs, snr_acc, cla_acc, s_mse = [AverageMeter() for _ in range(7)]
    metrics = [losses, psnrs, ms_ssims, snrs, snr_acc, cla_acc, s_mse]
    

    # awl = AutomaticWeightedLoss(1)
    # awl = AutomaticWeightedLoss(4)
    # optimizer = torch.optim.Adam([
    #             {'params': model.parameters()},
    #             {'params': awl.parameters(), 'weight_decay': 0}])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        if config.datasets == 'CIFAR10':
            images, label = data
        else:
            images = data
        images = images.cuda(non_blocking=True).float()
        label = label.cuda(non_blocking=True)
        out, cbr, g_snr, snr, cla, semantic_feature, x_signal, _, _ = model(images)
        psnr_loss = psnr_crit(out*255., images*255.)
        g_snr = torch.full((snr.size(0),), g_snr).cuda().float()
        snr_loss = snr_crit(g_snr, snr)
        snr_a = (d_snr(snr) == g_snr).float().mean()
        signal_loss = signal_crit(semantic_feature, x_signal)
        cla_loss = cla_crit(cla, label)
        cla_a = (cla.argmax(1) == label).float().mean()

        loss = 0.1*psnr_loss + 0.1*snr_loss + 10*signal_loss + 10*cla_loss
        # loss = awl(psnr_loss, snr_loss, signal_loss, cla_loss)
        # loss = awl(psnr_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        losses.update(loss.item())
        ms_ssims.update(1 - CalcuSSIM(images, out.clamp(0., 1.)).mean().item())   
        psnrs.update(psnr_loss_cal(out,images).item())
        snrs.update(g_snr[0].item())     
        snr_acc.update(snr_a.item())
        cla_acc.update(cla_a.item())
        s_mse.update(signal_loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info =(' | '.join([
                    f'Epoch {epoch}',
                    f'iter {iter}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbr}',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {ms_ssims.val:.3f} ({ms_ssims.avg:.3f})',
                    f'snr_acc {snr_acc.val:.3f} ({snr_acc.avg:.3f})',
                    f'cla_acc {cla_acc.val:.3f} ({cla_acc.avg:.3f})',
                    f'signal_loss {s_mse.val:.3f} ({s_mse.avg:.3f})',
                ]))
            print(log_info)
            logger.info(log_info)
            for i in metrics:
                i.clear()
    # scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    psnr_crit, 
                    snr_crit,
                    signal_crit,
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    multiple_snr = config.multiple_snr
    loss_list, psnr_list, ms_ssim_list, snr_acc_list, cla_acc_list, s_mse_list\
                                         =  [[] for _ in range(len(multiple_snr))], \
                                            [[] for _ in range(len(multiple_snr))], \
                                            [[] for _ in range(len(multiple_snr))], \
                                            [[] for _ in range(len(multiple_snr))], \
                                            [[] for _ in range(len(multiple_snr))], \
                                            [[] for _ in range(len(multiple_snr))]
    psnr_loss_cal = MSE().calculate_psnr
    if config.datasets == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for data in tqdm(test_loader):
                if config.datasets == 'CIFAR10':
                    img, label = data
                else:
                    img = data
                img = img.cuda(non_blocking=True).float()
                label = label.cuda(non_blocking=True)
                out, cbr, g_snr, snr, cla, semantic_feature, x_signal, _, _ = model(img,multiple_snr[i])
                g_snr = torch.full((snr.size(0),), g_snr).cuda()
                snr_loss = snr_crit(semantic_feature, x_signal)
                psnr_loss = psnr_crit(out*255., img*255.)
                loss = psnr_loss + 100*snr_loss
                snr_a = (d_snr(snr) == g_snr).float().mean()

                loss_list[i].append(loss.item())
                psnr_list[i].append(psnr_loss_cal(out, img).item())
                snr_acc_list[i].append(snr_a.item())
                cla_acc_list[i].append((cla.argmax(1) == label).float().mean().item())
                ms_ssim_list[i].append(1 - CalcuSSIM(img, out.clamp(0., 1.)).mean().item())
                s_mse_list[i].append(snr_loss.item())

        loss_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in loss_list]
        psnr_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in psnr_list]
        ms_ssim_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in ms_ssim_list]
        snr_acc_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in snr_acc_list]
        cla_acc_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in cla_acc_list]
        s_mse_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in s_mse_list]

        log_info = (f'val epoch: {epoch}, loss: {loss_list_avg}, cbr: {cbr}, psnr: {psnr_list_avg},'
                    f' avg{np.mean(psnr_list_avg)}, ms_ssim: {ms_ssim_list_avg}, snr_acc: {snr_acc_list_avg},'
                    f'avg{np.mean(snr_acc_list_avg)}, cla_acc: {cla_acc_list_avg}, s_mse: {s_mse_list_avg}')
        print(log_info)
        logger.info(log_info)
    
    return np.mean(psnr_list_avg)


def test_one_epoch( test_loader,
                    kodak_loader,
                    sc_model,
                    criterion,
                    config,
                    logger,
                    test_data_name=None):
    # switch to evaluate mode
    sc_model.eval()
    multiple_snr = config.multiple_snr
    s_mse_list, psnr_list, ms_ssim_list, time_list, snr_acc_list, cla_acc_list, mse_4_list, mse_1_list = \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))]

    psnr_loss_cal = MSE().calculate_psnr
    if config.datasets == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    

    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for j, data in enumerate(tqdm(test_loader)):
                imgs, labels = data
                imgs = imgs.cuda(non_blocking=True).float()
                labels = labels.cuda(non_blocking=True).float()
                start_time = time.time()
                out, cbr, g_snr, snr, cla, semantic_feature, x_signal, mse4, mse1 = sc_model(imgs,multiple_snr[i])
                snr_a = (d_snr(snr) == g_snr).float().mean()
                time_list[i].append(time.time()-start_time)
                psnr_list[i].append(psnr_loss_cal(out, imgs).item())
                ms_ssim_list[i].append(1 - CalcuSSIM(imgs, out.clamp(0., 1.)).mean().item())
                snr_acc_list[i].append(snr_a.item())
                cla_acc_list[i].append((cla.argmax(1) == labels).float().mean().item())
                s_mse_list[i].append(criterion(semantic_feature, x_signal).item())
                mse_4_list[i].append(mse4.item())
                mse_1_list[i].append(mse1.item())

        psnr_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in psnr_list]
        ms_ssim_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in ms_ssim_list]
        time_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in time_list]
        snr_acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in snr_acc_list]
        cla_acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in cla_acc_list]
        s_mse_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in s_mse_list]
        mse_4_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in mse_4_list]
        mse_1_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in mse_1_list]

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
        log_info = (f'test in CIFAR, cbr: {cbr}, psnr: {psnr_list_avg},avg{np.mean(psnr_list_avg)}, ms_ssim: {ms_ssim_list_avg}, '
                    f'cla_acc: {cla_acc_list_avg}, snr_acc: {snr_acc_list_avg}, avg{np.mean(snr_acc_list_avg)}'
                    f'cla_acc: {cla_acc_list_avg}, s_mse: {s_mse_list_avg}, time: {time_list_avg},'
                    f'mse_4: {mse_4_list_avg}, mse_1: {mse_1_list_avg}')
        print(log_info)
        logger.info(log_info)
    s_mse_list, psnr_list, ms_ssim_list, time_list, snr_acc_list, cla_acc_list, mse_4_list, mse_1_list = \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))]
    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for j, data in enumerate(tqdm(kodak_loader)):
                imgs = data
                imgs = imgs.cuda(non_blocking=True).float()
                start_time = time.time()
                out, cbr, g_snr, snr, cla, semantic_feature, x_signal, _, _ = sc_model(imgs,multiple_snr[i])
                snr_a = (torch.round(snr) == g_snr).float().mean()
                time_list[i].append(time.time()-start_time)
                psnr_list[i].append(psnr_loss_cal(out, imgs).item())
                ms_ssim_list[i].append(1 - CalcuSSIM(imgs, out.clamp(0., 1.)).mean().item())
                snr_acc_list[i].append(snr_a.item())
                s_mse_list[i].append(criterion(semantic_feature, x_signal).item())

                if config.test_datasets == 'Kodak':
                    save_imgs(imgs, out, j, config.work_dir + 'outputs/', test_data_name='_%ssnr'%multiple_snr[i])
                else:
                    if i == j:
                        save_imgs(imgs, out, i, config.work_dir + 'outputs/', test_data_name='_%ssnr'%multiple_snr[i])

        psnr_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in psnr_list]
        ms_ssim_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in ms_ssim_list]
        time_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in time_list]
        snr_acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in snr_acc_list]
        s_mse_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in s_mse_list]
        mse_4_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in mse_4_list]
        mse_1_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in mse_1_list]

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
        log_info = (f'test in kodak, cbr: {cbr}, psnr: {psnr_list_avg}, avg{np.mean(psnr_list_avg)} '
                    f'ms_ssim: {ms_ssim_list_avg}, snr_acc: {snr_acc_list_avg}, avg:{np.mean(snr_acc_list_avg)}'
                    f's_mse:{s_mse_list_avg}, avg:{np.mean(s_mse_list_avg)} time: {time_list_avg}'
                    f'mse_4: {mse_4_list_avg}, mse_1: {mse_1_list_avg}')
        print(log_info)
        logger.info(log_info)

    return np.mean(psnr_list_avg)


def test_one_epoch_old( val_loader,
                    test_loader,
                    sc_model,
                    cl_model,
                    criterion,
                    config,
                    logger,
                    test_data_name=None):
    # switch to evaluate mode
    sc_model.eval()
    cl_model.eval()
    multiple_snr = config.multiple_snr
    loss_list, psnr_list, ms_ssim_list, time_list, acc_list, snr_acc_list, cla_acc_list = \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
            [[] for _ in range(len(multiple_snr))], \
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
                    img,_ = data
                else:
                    img = data
                img  = img.cuda(non_blocking=True).float()
                start_time = time.time()
                out, cbr, g_snr, snr, cla = sc_model(img,multiple_snr[i])
                time_list[i].append(time.time()-start_time)

                loss = criterion(out*255., img*255.)
                loss_list[i].append(loss.item())
                psnr_list[i].append(psnr_loss(out, img).item())
                ms_ssim_list[i].append(1 - CalcuSSIM(img, out.clamp(0., 1.)).mean().item())

                if config.test_datasets == 'Kodak':
                    save_imgs(img, out, j, config.work_dir + 'outputs/', test_data_name='_%ssnr'%multiple_snr[i])
                else:
                    if i == j:
                        save_imgs(img, out, i, config.work_dir + 'outputs/', test_data_name='_%ssnr'%multiple_snr[i])
            for j, data in enumerate(tqdm(val_loader)):
                imgs, labels = data
                imgs = imgs.cuda(non_blocking=True).float()
                labels = labels.cuda(non_blocking=True).float()
                out, cbr, g_snr, snr, cla = sc_model(imgs,multiple_snr[i])
                preds = cl_model(out)
                test_acc = (preds.argmax(1) == labels).float().mean()
                acc_list[i].append(test_acc.item())
                snr_acc_list[i].append((snr.argmax(1) == g_snr).float().mean().item())
                cla_acc_list[i].append((cla.argmax(1) == labels).float().mean().item())

        loss_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in loss_list]
        psnr_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in psnr_list]
        ms_ssim_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in ms_ssim_list]
        time_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in time_list]
        acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in acc_list]
        snr_acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in snr_acc_list]
        cla_acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in cla_acc_list]

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
        log_info = (f'test of best model, loss: {loss_list_avg} , cbr: {cbr}, snr: {config.multiple_snr[g_snr]},'
                    f' psnr: {psnr_list_avg},avg{np.mean(psnr_list_avg)} ms_ssim: {ms_ssim_list_avg},acc:{acc_list_avg},cla_acc: {cla_acc_list_avg}'
                    f'snr_acc: {snr_acc_list_avg}, cla_acc: {cla_acc_list_avg},  time: {time_list_avg}')
        print(log_info)
        logger.info(log_info)

    return loss_list_avg, psnr_list_avg, ms_ssim_list_avg