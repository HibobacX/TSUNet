import os
from config import Config

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import network_parameters, SSIM
import random
import time
import numpy as np
import utils
from data_RGB import get_training_data, get_validation_data
from TSUnet import TSUNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
from torch.utils.tensorboard import SummaryWriter
import lpips

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

# result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

# utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = TSUNet(n_feat=74, scale_unetfeats=44, scale_orsnetfeats=32)
model_restoration.cuda()
parament_sum = network_parameters(model_restoration)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
# criterion_char = losses.CharbonnierLoss()
criterion_char = losses.CombinedLoss(lambda_vgg=1.3, lambda_l1=1.7)
criterion_char = criterion_char.cuda()
criterion_edge = losses.EdgeLoss().cuda()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')
# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(opt.TRAINING.TRAIN_PS) + 'x' + str(opt.TRAINING.TRAIN_PS)}
    Val patches size:   {str(opt.TRAINING.VAL_PS) + 'x' + str(opt.TRAINING.VAL_PS)}
    Model parameters:   {f"{parament_sum:.2f} M"}
    Start/End epochs:   {str(start_epoch) + '~' + str(opt.OPTIM.NUM_EPOCHS + 1)}
    Batch sizes:        {opt.OPTIM.BATCH_SIZE}
    Learning rate:      {opt.OPTIM.LR_INITIAL}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

best_psnr = 0
best_ssim = 0
best_lpips = 1
best_epoch = 0
lpips_fn = lpips.LPIPS(net='alex')  # 使用AlexNet
lpips_fn.cuda()  # 将模型移动到GPU

total_start_time = time.time()
##############log##############
log_dir = os.path.join(model_dir, 'log')
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        #         print(target.shape)

        restored = model_restoration(input_)

        # Compute loss at each stage
        loss_char = torch.sum(torch.stack([criterion_char(restored[j], target) for j in range(len(restored))]))

        loss_edge = torch.sum(torch.stack([criterion_edge(restored[j], target) for j in range(len(restored))]))
        loss = loss_char + (0.05 * loss_edge)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        lpips_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)
            restored = restored[0]

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(restored, target))
                lpips_val_rgb.append(lpips_fn(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
        lpips_val_rgb = torch.stack(lpips_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        # 保存最好的LPIPS模型
        if lpips_val_rgb < best_lpips:
            best_lpips = lpips_val_rgb
            best_epoch_lpips = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestLPIPS.pth"))

        print("[epoch %d LPIPS: %.4f --- best_epoch %d Best_LPIPS %.4f]" % (epoch, lpips_val_rgb, best_epoch_lpips, best_lpips))


        writer.add_scalar('PSNR/Val', psnr_val_rgb, epoch)
        writer.add_scalar('SSIM/Val', ssim_val_rgb, epoch)
        writer.add_scalar('LPIPS/Val', lpips_val_rgb, epoch)

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Learning_Rate', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
