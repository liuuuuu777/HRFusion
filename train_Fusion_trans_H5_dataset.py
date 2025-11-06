# -*- coding: utf-8 -*- 
# @Time : 2024/10/19 10:33 
# @Author : bfliu Beijing University of Posts and Telecommunications
# @File : train_Fusion.py
# @contact: bfliu@bupt.edu.cn

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from colorama import Fore, Style
import scipy.io as scio

from nets.FusionNet import Restormer_Encoder, Fusion_Decoder
from nets.generate_model import ResnetGenerator_Encoder_decoder
from Dataset_loader import H5Dataset
from utils import Transformer
from nets.loss import EdgeLoss, Gradient_loss, set_requires_grad

"""
------------------------------------------------------------------------------
Environment Settings
------------------------------------------------------------------------------
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

"""
------------------------------------------------------------------------------
Training HyperParameters
------------------------------------------------------------------------------
"""
batch_size = 4
num_epochs = 10
lr = 1e-3
weight_decay = 0
optim_step = 1
optim_gamma = 0.5

"""
------------------------------------------------------------------------------
Loss Function Coefficient
------------------------------------------------------------------------------
"""
coeff_pixel = 10
coeff_grad = 100

"""
------------------------------------------------------------------------------
Save Format Settings
------------------------------------------------------------------------------
"""
save_place = "./train_results/"
data_path = "C:/Users/root/Desktop/LLVIP_no_reverse_120_128_200.h5"

"""
------------------------------------------------------------------------------
Build Model
------------------------------------------------------------------------------
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

I_generate_Encoder = ResnetGenerator_Encoder_decoder().to(device)
I_extract_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
V_generate_Encoder = ResnetGenerator_Encoder_decoder().to(device)
V_extract_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder_Fusion = nn.DataParallel(Fusion_Decoder()).to(device)

I_generate_Encoder.load_state_dict(
    torch.load("model/generate/256_128_64_32_down1connect_C8_16_32_64/latest_net_G_A.pth", map_location=device))
I_extract_Encoder.load_state_dict(
    torch.load("model/extract/C8_16_32_64/Encoder_ir.model", map_location=device).state_dict())
V_generate_Encoder.load_state_dict(
    torch.load("model/generate/256_128_64_32_down1connect_C8_16_32_64/latest_net_G_B.pth", map_location=device))
V_extract_Encoder.load_state_dict(
    torch.load("model/extract/C8_16_32_64/Encoder_vi.model", map_location=device).state_dict())

resume = False
ep = "0"

if resume:
    lr = optim_gamma ** (-int(ep) - 1) * lr
    Decoder_Fusion.load_state_dict(
        torch.load(os.path.join(save_place, "epoch_{}".format(ep), "Decoder_Fusion.model")).state_dict())

"""
------------------------------------------------------------------------------
Loss Function
------------------------------------------------------------------------------
"""
Edge_Loss = EdgeLoss()
Loss_grad = Gradient_loss(1).to(device)

"""
------------------------------------------------------------------------------
Optimizer and Scheduler
------------------------------------------------------------------------------
"""
optimizer = torch.optim.Adam(Decoder_Fusion.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

"""
------------------------------------------------------------------------------
DataSet and DataLoader
------------------------------------------------------------------------------
"""
trainloader = DataLoader(H5Dataset(data_path), batch_size=batch_size, shuffle=True, num_workers=0)

"""
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
"""
torch.backends.cudnn.benchmark = True

I_generate_Encoder.eval()
I_extract_Encoder.eval()
V_generate_Encoder.eval()
V_extract_Encoder.eval()
Decoder_Fusion.train()

total_loss = []
total_loss_pixel = []
total_loss_grad = []
total_loss_edge = []

trans = Transformer(shift_n=2, rotate_n=3, flip_n=2)

for epoch in range(int(ep), num_epochs):
    save_place_dir = os.path.join(save_place, "epoch_{}".format(epoch + 1))
    save_place_dir_loss = os.path.join(save_place_dir, "loss", "epoch_{}".format(epoch + 1))

    if not os.path.exists(save_place_dir_loss):
        os.makedirs(save_place_dir_loss)

    for i, (img_IR, img_VI, img_SOTA) in enumerate(trainloader):
        img_IR, img_VI, img_SOTA = trans.apply(img_IR, img_VI, img_SOTA)
        img_IR, img_VI, img_SOTA = img_IR.cuda(), img_VI.cuda(), img_SOTA.cuda()

        optimizer.zero_grad()

        Feature_I_generate = I_generate_Encoder(img_IR)
        Feature_I_extract = I_extract_Encoder(img_IR)
        Feature_V_generate = V_generate_Encoder(img_VI)
        Feature_V_extract = V_extract_Encoder(img_VI)

        set_requires_grad(Decoder_Fusion, True)
        img_Fu = Decoder_Fusion(Feature_I_generate, Feature_I_extract, Feature_V_generate, Feature_V_extract)

        x_in_max = torch.max(img_IR, img_VI)
        loss_pixel = F.l1_loss(x_in_max, img_Fu)
        loss_edge = Edge_Loss(img_Fu, img_SOTA)
        loss_grad = Loss_grad(img_Fu, img_IR, img_VI)

        loss_total = coeff_pixel * loss_pixel + coeff_grad * loss_grad

        loss_total.backward()
        optimizer.step()

        if i % 100 == 0:
            mesg = "{}\tepoch {}/{}\t {}/{} \ttotal_loss {}\tloss_grad {}\tloss_edge {}\tloss_pixel {}\n".format(
                time.ctime(), epoch + 1, num_epochs, i, len(trainloader),
                              Fore.RED + str(loss_total.item()) + Style.RESET_ALL,
                              Fore.GREEN + str(loss_grad.item()) + Style.RESET_ALL,
                              Fore.YELLOW + str(loss_edge.item()) + Style.RESET_ALL,
                loss_pixel.item()
            )
            print(mesg)

            total_loss.append(loss_total.item())
            total_loss_grad.append(loss_grad.item())
            total_loss_pixel.append(loss_pixel.item())
            total_loss_edge.append(loss_edge.item())

    # Save model and losses
    Decoder_Fusion.eval()
    Decoder_Fusion.cpu()
    torch.save(Decoder_Fusion, os.path.join(save_place_dir, "Decoder_Fusion.model"))

    scio.savemat(os.path.join(save_place_dir_loss, 'total_loss.mat'), {'total_loss': total_loss})
    scio.savemat(os.path.join(save_place_dir_loss, 'total_loss_grad.mat'), {'total_loss_grad': total_loss_grad})
    scio.savemat(os.path.join(save_place_dir_loss, 'total_loss_pixel.mat'), {'total_loss_pixel': total_loss_pixel})

    Decoder_Fusion.train()
    Decoder_Fusion.cuda()

    print("\nCheckpoint, trained model saved at: " + save_place_dir)

    scheduler.step()
    if optimizer.param_groups[0]['lr'] <= 1e-8:
        optimizer.param_groups[0]['lr'] = 1e-8

print("\nDone.")
