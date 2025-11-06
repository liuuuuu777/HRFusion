# -*- coding: utf-8 -*- 
# @Time : 2024/10/15 15:52
# @Author : bfliu Beijing University of Posts and Telecommunications
# @File : train_TransFE.py
# @contact: bfliu@bupt.edu.cn

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import scipy.io as scio

from nets.FusionNet import Restormer_Encoder, Restormer_Decoder
from dataprocessing.SOTA_DATASET import KAIST_LLVIPDataset
from nets.loss import loss_fusion

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
num_epochs = 5
lr = 1e-3
weight_decay = 0
optim_step = 1
optim_gamma = 0.1

"""
------------------------------------------------------------------------------
Save Format Settings
------------------------------------------------------------------------------
"""
save_place = "./train_results/encoder_vi/img_encoder_ir_C8_16_32_64_transformer"
llvip = "/data/data3/Users/root/Desktop/IVIFdata/LLVIP/LLVIP/infrared/train"

"""
------------------------------------------------------------------------------
Build Model
------------------------------------------------------------------------------
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder = nn.DataParallel(Restormer_Decoder()).to(device)

"""
------------------------------------------------------------------------------
Optimizer and Scheduler
------------------------------------------------------------------------------
"""
optimizer1 = torch.optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Decoder.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)

"""
------------------------------------------------------------------------------
DataSet and DataLoader
------------------------------------------------------------------------------
"""
trainloader = DataLoader(
    KAIST_LLVIPDataset(llvip),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

"""
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
"""
torch.backends.cudnn.benchmark = True
Encoder.train()
Decoder.train()

print("seed:", torch.seed())

loss_fn = loss_fusion()

total_loss = []
total_loss_pixel = []

for epoch in range(num_epochs):
    for i, (_, img, _) in enumerate(trainloader):
        img = img.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        Feature_img = Encoder(img)
        output = Decoder(Feature_img)

        # Calculate loss
        loss_pixel = loss_fn(output, img)
        loss_total = loss_pixel

        loss_total.backward()
        optimizer1.step()
        optimizer2.step()

        if i % 100 == 0:
            mesg = "{}\tepoch {}/{}\tbatch {}\tloss_pixel {:.4f}\tloss_total {:.4f}".format(
                time.ctime(), epoch + 1, num_epochs, i, loss_pixel.item(), loss_total.item()
            )
            print(mesg)

        total_loss.append(loss_total.item())
        total_loss_pixel.append(loss_pixel.item())

        # Clean up memory
        del img, Feature_img, output, loss_total, loss_pixel
        torch.cuda.empty_cache()
        gc.collect()

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_place_dir = os.path.join(save_place, "epoch_{}".format(epoch + 1))
        loss_dir = os.path.join(save_place_dir, "loss")

        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)

        Encoder_save = os.path.join(save_place_dir, "Encoder.model")
        Decoder_save = os.path.join(save_place_dir, "Decoder.model")

        Encoder.eval()
        Encoder.cpu()
        Decoder.eval()
        Decoder.cpu()

        torch.save(Encoder, Encoder_save)
        torch.save(Decoder, Decoder_save)

        # Save loss at final epoch
        if epoch == num_epochs - 1:
            scio.savemat(os.path.join(loss_dir, 'total_loss.mat'), {'total_loss': total_loss})
            scio.savemat(os.path.join(loss_dir, 'total_loss_pixel.mat'), {'total_loss_pixel': total_loss_pixel})

        Encoder.train()
        Encoder.cuda()
        Decoder.train()
        Decoder.cuda()

        print("\nCheckpoint, trained model saved at: " + save_place_dir)

    scheduler1.step()
    scheduler2.step()

print("\nDone.")
