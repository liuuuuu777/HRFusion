# -*- coding: utf-8 -*- 
# @Time : 2024/10/17 21:29 
# @Author : bfliu Beijing University of Posts and Telecommunications
# @File : loss.py
# @contact: bfliu@bupt.edu.cn


import torch
import torch.nn as nn

import torchvision
import torch.nn.functional as F
import kornia

class loss_fusion(nn.Module):
    def __init__(self, coeff_int=1, coeff_grad=1):
        super(loss_fusion, self).__init__()
        self.coeff_int = coeff_int
        self.coeff_grad = coeff_grad

    def forward(self, pre, target):
        loss_int = F.l1_loss(pre, target)
        loss_grad = F.l1_loss(kornia.filters.SpatialGradient()(pre), kornia.filters.SpatialGradient()(target))

        loss_total = self.coeff_int * loss_int + self.coeff_grad * loss_grad
        return loss_total


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        L2_distances = torch.cdist(X_flat, X_flat) ** 2
        rbf_matrix = torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        )
        return rbf_matrix.sum(dim=0)


class PoliKernel(nn.Module):
    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        K = (torch.matmul(X_flat, X_flat.t()) + self.constant_term) ** self.degree
        return K


class LinearKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        K = torch.matmul(X_flat, X_flat.t())
        return K


class LaplaceKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        L2_distances = torch.cdist(X_flat, X_flat) ** 2
        laplace_matrix = torch.exp(
            -L2_distances[None, ...] * (self.gammas)[:, None, None]
        )
        return laplace_matrix.sum(dim=0)


class M3DLoss(nn.Module):
    def __init__(self, kernel_type):
        super().__init__()
        if kernel_type == "gaussian":
            self.kernel = RBF()
        elif kernel_type == "linear":
            self.kernel = LinearKernel()
        elif kernel_type == "polinominal":
            self.kernel = PoliKernel()
        elif kernel_type == "laplace":
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # 计算核矩阵
        K = self.kernel(torch.cat([X_flat, Y_flat], dim=0))

        X_size = X.size(0)
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        return abs(XX - 2 * XY + YY)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

import numpy as np
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
# class Gradient_loss(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         # average
#         patch_szie = 3
#         reflection_padding = int(np.floor(patch_szie / 2))
#         weight = torch.ones([1, 1, patch_szie, patch_szie])
#         self.conv_avg = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
#                                   stride=1, padding=reflection_padding, bias=False)
#         self.conv_avg.weight.data = (1 / (patch_szie * patch_szie)) * weight.repeat(channels, channels, 1, 1).float()
#         self.conv_avg.requires_grad_(False)
#
#         # self.conv_one = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
#         weight = torch.from_numpy(np.array([[[[0., 1., 0.],
#                                               [1., -4., 1.],
#                                               [0., 1., 0.]]]]))
#         self.conv_two = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
#         self.conv_two.weight.data = weight.repeat(channels, channels, 1, 1).float()
#         self.conv_two.requires_grad_(False)
#
#         # LoG
#         weight_log = torch.from_numpy(np.array([[[[0., 0., -1, 0., 0.],
#                                                   [0., -1, -2, -1, 0.],
#                                                   [-1, -2, 16, -2, -1],
#                                                   [0., -1, -2, -1, 0.],
#                                                   [0., 0., -1, 0., 0.]]]]))
#         # weight_log = torch.from_numpy(np.array([[[[0., 0., 0., -1, 0., 0., 0.],
#         #                                           [0., 0., -1, -2, -1, 0., 0.],
#         #                                           [0., -1, -2, -3, -2, -1, 0.],
#         #                                           [-1, -2, -3, 40, -3, -2, -1],
#         #                                           [0., -1, -2, -3, -2, -1, 0.],
#         #                                           [0., 0., -1, -2, -1, 0., 0.],
#         #                                           [0., 0., 0., -1, 0., 0., 0.]]]]))
#         self.conv_log = torch.nn.Conv2d(channels, channels, (5, 5), stride=1, padding=3, bias=False)
#         self.conv_log.weight.data = weight_log.repeat(channels, channels, 1, 1).float()
#         self.conv_log.requires_grad_(False)
#
#         # sobel
#         weight_s1 = torch.from_numpy(np.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]))
#         weight_s2 = torch.from_numpy(np.array([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]]))
#         self.conv_sx = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
#         self.conv_sx.weight.data = weight_s2.repeat(channels, channels, 1, 1).float()
#         self.conv_sx.requires_grad_(False)
#         self.conv_sy = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
#         self.conv_sy.weight.data = weight_s1.repeat(channels, channels, 1, 1).float()
#         self.conv_sy.requires_grad_(False)
#
#         # average
#         patch_szie = 3
#         reflection_padding = int(np.floor(patch_szie / 2))
#         weight_avg = torch.ones([1, 1, patch_szie, patch_szie])
#         self.conv_avg = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
#                                   stride=1, padding=reflection_padding, bias=False)
#         self.conv_avg.weight.data = (1 / (patch_szie * patch_szie)) * weight_avg.repeat(channels, channels, 1,
#                                                                                         1).float()
#         self.conv_avg.requires_grad_(False)
#         self.sobelconv = Sobelxy()
#
#     def forward(self, out, x_ir, x_vi):
#         channels = x_ir.size()[1]
#         channels_t = out.size()[1]
#         assert channels == channels_t, \
#             f"The channels of x ({channels}) doesn't match the channels of target ({channels_t})."
#         g_o = torch.clamp(self.conv_two(out), min=0)
#         g_xir = torch.clamp(self.conv_two(x_ir), min=0)
#         g_xvi = torch.clamp(self.conv_two(x_vi), min=0)
#         g_sub = g_o
#
#         g_target = torch.max(g_xir, g_xvi)
#         loss = l1_loss(g_sub, g_target)
#
#         return loss

class Gradient_loss(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # average
        patch_szie = 3
        reflection_padding = int(np.floor(patch_szie / 2))
        weight = torch.ones([1, 1, patch_szie, patch_szie])
        self.conv_avg = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
                                  stride=1, padding=reflection_padding, bias=False)
        self.conv_avg.weight.data = (1 / (patch_szie * patch_szie)) * weight.repeat(channels, channels, 1, 1).float()
        self.conv_avg.requires_grad_(False)

        # self.conv_one = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        weight = torch.from_numpy(np.array([[[[0., 1., 0.],
                                              [1., -4., 1.],
                                              [0., 1., 0.]]]]))
        self.conv_two = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_two.weight.data = weight.repeat(channels, channels, 1, 1).float()
        self.conv_two.requires_grad_(False)

        # LoG
        weight_log = torch.from_numpy(np.array([[[[0., 0., -1, 0., 0.],
                                                  [0., -1, -2, -1, 0.],
                                                  [-1, -2, 16, -2, -1],
                                                  [0., -1, -2, -1, 0.],
                                                  [0., 0., -1, 0., 0.]]]]))
        # weight_log = torch.from_numpy(np.array([[[[0., 0., 0., -1, 0., 0., 0.],
        #                                           [0., 0., -1, -2, -1, 0., 0.],
        #                                           [0., -1, -2, -3, -2, -1, 0.],
        #                                           [-1, -2, -3, 40, -3, -2, -1],
        #                                           [0., -1, -2, -3, -2, -1, 0.],
        #                                           [0., 0., -1, -2, -1, 0., 0.],
        #                                           [0., 0., 0., -1, 0., 0., 0.]]]]))
        self.conv_log = torch.nn.Conv2d(channels, channels, (5, 5), stride=1, padding=3, bias=False)
        self.conv_log.weight.data = weight_log.repeat(channels, channels, 1, 1).float()
        self.conv_log.requires_grad_(False)

        # sobel
        weight_s1 = torch.from_numpy(np.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]))
        weight_s2 = torch.from_numpy(np.array([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]]))
        self.conv_sx = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_sx.weight.data = weight_s2.repeat(channels, channels, 1, 1).float()
        self.conv_sx.requires_grad_(False)
        self.conv_sy = torch.nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_sy.weight.data = weight_s1.repeat(channels, channels, 1, 1).float()
        self.conv_sy.requires_grad_(False)

        # average
        patch_szie = 3
        reflection_padding = int(np.floor(patch_szie / 2))
        weight_avg = torch.ones([1, 1, patch_szie, patch_szie])
        self.conv_avg = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
                                  stride=1, padding=reflection_padding, bias=False)
        self.conv_avg.weight.data = (1 / (patch_szie * patch_szie)) * weight_avg.repeat(channels, channels, 1,
                                                                                        1).float()
        self.conv_avg.requires_grad_(False)
        self.sobelconv = Sobelxy()

    def forward(self, out, x_ir, x_vi):
        channels = x_ir.size()[1]
        channels_t = out.size()[1]
        assert channels == channels_t, \
            f"The channels of x ({channels}) doesn't match the channels of target ({channels_t})."
        y_grad = self.sobelconv(x_vi)
        ir_grad = self.sobelconv(x_ir)
        generate_img_grad = self.sobelconv(out)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        return loss_grad


class Order_loss(nn.Module):
    def __init__(self, channels, patch_szie=11):
        super().__init__()

        reflection_padding = int(np.floor(patch_szie / 2))
        weight = torch.ones([1, 1, patch_szie, patch_szie])
        self.conv_two = nn.Conv2d(channels, channels, (patch_szie, patch_szie),
                                  stride=1, padding=reflection_padding, bias=False)
        self.conv_two.weight.data = (1 / (patch_szie * patch_szie)) * weight.repeat(channels, channels, 1, 1).float()
        self.conv_two.requires_grad_(False)

        # LoG
        weight_log = torch.from_numpy(np.array([[[[0., 0., -1, 0., 0.],
                                                  [0., -1, -2, -1, 0.],
                                                  [-1, -2, 16, -2, -1],
                                                  [0., -1, -2, -1, 0.],
                                                  [0., 0., -1, 0., 0.]]]]))
        self.conv_log = torch.nn.Conv2d(channels, channels, (5, 5), stride=1, padding=2, bias=False)
        self.conv_log.weight.data = weight_log.repeat(channels, channels, 1, 1).float()
        self.conv_log.requires_grad_(False)

    def forward(self, out, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = torch.sqrt((x - self.conv_two(x)) ** 2)
        # g_y = torch.sqrt((y - self.conv_two(y)) ** 2)
        # LoG
        # i_x = self.conv_two(x)
        # i_x = (i_x - torch.min(i_x)) / (torch.max(i_x) - torch.min(i_x))
        # g_x = torch.clamp(self.conv_log(x), min = 0)
        # g_x = (g_x - torch.min(g_x)) / (torch.max(g_x) - torch.min(g_x))
        # s_x = 0.4 * i_x + 0.6 * g_x

        # i_y = self.conv_two(y)
        # i_y = (i_y - torch.min(i_y)) / (torch.max(i_y) - torch.min(i_y))
        # g_y = torch.clamp(self.conv_log(y), min = 0)
        # g_y = (g_y - torch.min(g_y)) / (torch.max(g_y) - torch.min(g_y))
        # s_y = 0.6 * i_y + 0.4 * g_y
        # Avg
        EPSILON = 1e-6

        s_x = self.conv_two(x)
        # s_x = (s_x - torch.min(s_x)) / (torch.max(s_x) - torch.min(s_x))
        s_y = self.conv_two(y)
        # s_y = (s_y - torch.min(s_y)) / (torch.max(s_y) - torch.min(s_y))
        w_x = s_x / (s_x + s_y + EPSILON)
        w_y = s_y / (s_x + s_y + EPSILON)

        # target = torch.max((w_x + 1) * x, (w_y + 1) * y)
        t_one = torch.ones_like(w_x)
        mask = torch.clamp(w_x - w_y, min=0.0)
        mask = torch.where(mask > 0, t_one, mask)

        target = mask * x + (1 - mask) * y
        loss_p = l1_loss(out, target)

        return loss_p




class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def set_requires_grad( nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.gram = Gram()
        if torch.cuda.is_available():
            self.vgg.cuda()
            self.gram.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        contentloss = 0
        styleloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        for i in range(len(x_vgg)):
            styleloss += self.weights[i] * self.criterion2(self.gram(x_vgg[i]), self.gram(y_vgg[i].detach()))
        contentloss = self.L1Loss(x_vgg[3], y_vgg[3].detach())

        allloss = self.L1Loss(x,y) + contentloss + 100 * styleloss
        return allloss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss