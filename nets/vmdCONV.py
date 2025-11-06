
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import fft2,ifft2
from PIL import Image


EPOS = 1e+6


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, padding_mode="reflect"):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)

    def forward(self, x):
        real = x.real
        imag = x.imag
        return torch.complex(
            self.conv_real(real), # - self.conv_imag(imag)
            self.conv_real(imag) #+ self.conv_imag(real)
        )


class ComplexInstanceNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexInstanceNorm2d, self).__init__()
        self.in_real = nn.InstanceNorm2d(num_features)
        self.in_imag = nn.InstanceNorm2d(num_features)

    def forward(self, x):
        return torch.complex(
            self.in_real(x.real),
            self.in_imag(x.imag)
        )


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.complex(
            self.relu(x.real),
            self.relu(x.imag)
        )

class ComplexLoss(nn.Module):
    def __init__(self):
        super(ComplexLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        return torch.abs(
            self.loss(x.real, y.real)+
            self.loss(x.imag, y.imag)
        )
class init_vmdconv(nn.Module):
    def __init__(self, channel):
        super(init_vmdconv, self).__init__()

        # Define complex convolutional blocks
        self.f1 = nn.Sequential(
            ComplexConv2d(channel, channel, kernel_size=3),
            ComplexInstanceNorm2d(channel),
            ComplexReLU()
        )

        self.f2 = nn.Sequential(
            ComplexConv2d(channel, channel, kernel_size=3),
            ComplexInstanceNorm2d(channel),
            ComplexReLU()
        )

        self.u_optim = nn.Sequential(
            ComplexConv2d(channel, channel, kernel_size=3),
            ComplexInstanceNorm2d(channel),
            ComplexReLU()
        )

        self.w2 = nn.Sequential(
            ComplexConv2d(channel, channel, kernel_size=3),
            ComplexInstanceNorm2d(channel),
            ComplexReLU()
        )

        self.u1s = []
        self.u2s = []

    def forward(self, x,  num=2):
        device = x.device
        dtype = x.dtype

        # Convert to numpy for FFT
        x_np = x.cpu().detach().numpy()

        # Apply FFT2 and convert back to tensor
        f_np = np.stack([fft2(x_np[:, i, :, :]) for i in range(x_np.shape[1])], axis=1)
        w = np.abs(f_np)

        # Convert to torch tensors
        f = torch.from_numpy(f_np).to(device).to(torch.complex64)
        w = torch.from_numpy(w).to(device).to(dtype)


        # Clear previous iterations
        self.u1s = []
        self.u2s = []

        # Initial decomposition
        u21 = self.f1(f)
        u11 = self.f2(f)

        self.u1s.append(u11)
        self.u2s.append(u21)

        # Convert absolute value to tensor
        w21 = torch.abs(u21)
        optim = f - self.u_optim(u21)

        # Calculate initial decompositions
        U22 = (f - u21 + optim / 2) / (1 + 2 * (w - w21) ** 2 + 1e-8)
        w12 = torch.abs(self.w2(U22))
        U12 = (f - U22 + optim / 2) / (1 + 2 * (w - w12) ** 2 + 1e-8)

        # Iterative refinement
        for i in range(num):
            self.u1s.append(U12)
            self.u2s.append(U22)

            # Calculate updated weights
            u2s_sum = sum(self.u2s)
            u2s_abs_sq = torch.abs(u2s_sum) ** 2
            w2 = torch.sum(w * u2s_abs_sq) / (torch.sum(u2s_abs_sq) + 1e-8)
            w1 = torch.abs(sum(self.u1s))

            # Update decompositions
            U12 = (f - U22 + optim) / (1 + 2 * (w - w1) ** 2 + 1e-8)
            U22 = (f - U12 + optim) / (1 + 2 * (w - w2) ** 2 + 1e-8)

            self.u1s.append(U12)
            self.u2s.append(U22)


        U1 = U12
        U2 = U22

        # Convert final output back to real domain
        # U1_ifft = torch.stack([torch.from_numpy(np.fft.ifft2(U12[:, i, :, :].cpu().detach().numpy()).real)
        #                        for i in range(U12.shape[1])], dim=1).to(device)
        # U2_ifft = torch.stack([torch.from_numpy(np.fft.ifft2(U22[:, i, :, :].cpu().detach().numpy()).real)
        #                        for i in range(U22.shape[1])], dim=1).to(device)

        return U1, U2




class vmd(nn.Module):
    def __init__(self,FFT_num):
        super(vmd, self).__init__()
        self.extract_channels = [8, 16, 32, 64]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extract_4 =nn.ModuleList([init_vmdconv(self.extract_channels[3]).to(device) for i in range(FFT_num)])

        self.generate_4 = nn.ModuleList([init_vmdconv(self.extract_channels[3]).to(device)  for i in range(FFT_num)])

        self.complexLoss = ComplexLoss()


    def forward(self, generate, extract):
        # IR/VI

        e4 = extract[3]
        g4 = generate[3]
        device = generate[3].device
        for i in range(len(self.extract_4)):

            extract_4_u1, extract_4_u2 = self.extract_4[i](e4)

            # generate_3_u1, generate3_u2 = self.generate_3(generate[2])
            generate_4_u1, generate_4_u2 = self.generate_4[i](g4)

            extract_4_u1 -= generate_4_u1
            extract_4_u2 -= generate_4_u2

            extract_4 = extract_4_u1 + extract_4_u2
            e4 = torch.stack([torch.from_numpy(np.fft.ifft2(extract_4[:, i, :, :].cpu().detach().numpy()).real)
                                                 for i in range(extract_4.shape[1])], dim=1).to(device).float()

        extract[3] = e4.float()
        l_loss = 0
        return extract , l_loss




