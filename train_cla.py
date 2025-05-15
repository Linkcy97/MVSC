import sys
import os
# 获取主程序文件的目录
main_dir = os.path.dirname(os.path.abspath(__file__))
# 将主程序目录添加到 sys.path
if main_dir not in sys.path:
    sys.path.append(main_dir)
import torch
from utils.datasets import get_loader
from tensorboardX import SummaryWriter
from models.mamba_vision import MVSC
from models.cla import Cla_Net
from models.djscc import Djscc
import os
from models.distortion import *
import sys
from tqdm import tqdm
from utils.utils import *
from configs.config import setting_config
from models.AutomaticWeightedLoss import AutomaticWeightedLoss

import warnings
warnings.filterwarnings("ignore")


def train_one_epoch(train_loader,
                    model,
                    cl_model,
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
    cl_model.train() 
    # 冻结模型的所有参数
    for param in model.parameters():
        param.requires_grad = False
    psnr_loss_cal = MSE().calculate_psnr
    losses, cla_acc = [AverageMeter() for _ in range(2)]
    metrics = [losses, cla_acc]

    # awl = AutomaticWeightedLoss(4)
    awl = AutomaticWeightedLoss(1)

    optimizer = torch.optim.Adam([
                {'params': cl_model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0}])

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        if config.datasets == 'CIFAR10':
            images, label = data
        else:
            images = data
        images = images.cuda(non_blocking=True).float()
        label = label.cuda(non_blocking=True)
        out, _, snr, _, _, _, _, _ ,_ = model(images)
        cla = cl_model(out)

        cla_loss = cla_crit(cla, label)
        cla_a = (cla.argmax(1) == label).float().mean()

        loss = awl(cla_loss)
        # loss = awl(psnr_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item())

        cla_acc.update(cla_a.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info =(' | '.join([
                    f'Epoch {epoch}',
                    f'iter {iter}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'cla_acc {cla_acc.val:.3f} ({cla_acc.avg:.3f})'
                    f'psnr_loss {psnr_loss_cal(out,images).item():.3f}',
                    f'snr {snr:.3f}'
                ]))
            print(log_info)
            logger.info(log_info)
            for i in metrics:
                i.clear()
    # scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    cl_model,
                    psnr_crit, 
                    snr_crit,
                    signal_crit,
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    multiple_snr = config.multiple_snr
    loss_list, cla_acc_list =  [[] for _ in range(len(multiple_snr))], \
                                            [[] for _ in range(len(multiple_snr))], \

    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for data in tqdm(test_loader):
                if config.datasets == 'CIFAR10':
                    img, label = data
                else:
                    img = data
                img = img.cuda(non_blocking=True).float()
                label = label.cuda(non_blocking=True)
                out, _, _, _, _, _, _, _, _ = model(img,multiple_snr[i])
                cla = cl_model(out)
                cla_acc_list[i].append((cla.argmax(1) == label).float().mean().item())


        loss_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in loss_list]
        cla_acc_list_avg = [round(np.mean(sublist),3) if sublist else 0 for sublist in cla_acc_list]


        log_info = (f'val epoch: {epoch}, loss: {loss_list_avg},'
                    f', cla_acc: {cla_acc_list_avg},')
        print(log_info)
        logger.info(log_info)
    
    return np.mean(cla_acc_list_avg)


def test_one_epoch( test_loader,
                    kodak_loader,
                    sc_model,
                    cl_model,
                    criterion,
                    config,
                    logger,
                    test_data_name=None):
    # switch to evaluate mode
    sc_model.eval()
    multiple_snr = config.multiple_snr
    cla_acc_list = [[] for _ in range(len(multiple_snr))]

    with torch.no_grad():
        for i in range(len(multiple_snr)):
            for j, data in enumerate(tqdm(test_loader)):
                imgs, labels = data
                imgs = imgs.cuda(non_blocking=True).float()
                labels = labels.cuda(non_blocking=True).float()
                out, _, _, _, _, _, _, _, _ = sc_model(imgs,multiple_snr[i])
                cla = cl_model(out)
                cla_acc_list[i].append((cla.argmax(1) == labels).float().mean().item())

        cla_acc_list_avg = [round(np.mean(sublist),4) if sublist else 0 for sublist in cla_acc_list]


        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
        log_info = (f'cla_acc: {cla_acc_list_avg}, '
                    f'cla_acc: {cla_acc_list_avg},' )
        print(log_info)
        logger.info(log_info)

    return np.mean(cla_acc_list_avg)

def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log_cla')
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

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()


    print('#----------Preparing dataset----------#')
    train_loader, val_loader, test_loader, kodak_loader = get_loader(config)

    print('#----------Prepareing Model----------#')
    model = MVSC(config)
    # model = Djscc(config)
    model = model.cuda()
    best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(best_weight,strict=False)

    cl_model = Cla_Net(3)
    cl_model = cl_model.cuda()

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

    step = 0
    print('#----------Training----------#')
    if not os.path.exists(os.path.join(checkpoint_dir, 'cla_best.pth')):
        for epoch in range(start_epoch, config.epochs + 1):

            torch.cuda.empty_cache()

            step = train_one_epoch(
                train_loader,
                model,
                cl_model,
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
                    cl_model,
                    psnr_crit,
                    snr_crit,
                    signal_crit,
                    epoch,
                    logger,
                    config
                )

            if score > max_score:
                torch.save(cl_model.state_dict(), os.path.join(checkpoint_dir, 'cla_best.pth'))
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
                    'model_state_dict': cl_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'cla_best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/cla_best.pth', map_location=torch.device('cpu'))
        cl_model.load_state_dict(best_weight)
        loss = test_one_epoch(  test_loader,
                                kodak_loader,
                                model,
                                cl_model,
                                psnr_crit,
                                config,
                                logger)
        os.rename(
            os.path.join(checkpoint_dir, 'cla_best.pth'),
            os.path.join(checkpoint_dir, f'cla_best-epoch{min_epoch}-max_score{max_score:.4f}.pth')
        )      
        logger.info(f'cla_best-epoch{min_epoch}-max_score{max_score:.4f}.pth')

if __name__ == '__main__':
    config = setting_config
    main(config)