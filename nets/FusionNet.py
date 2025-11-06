# -*- coding: utf-8 -*- 
# @Time : 2024/10/19 11:02 
# @Author : bfliu Beijing University of Posts and Telecommunications
# @File : FusionNet.py
# @contact: bfliu@bupt.edu.cn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers
from nets.vmdCONV import vmd as vmd



class Restormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads=8)
        self.salientFeature = salientFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
                             padding_mode="reflect")

    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)
        x2 = self.salientFeature(x)
        out = self.FFN(torch.cat((x1, x2), 1))
        return out


class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class salientFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,
                 ):
        super(salientFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[Dense_Block(2, dim, dim) for i in range(num_blocks)])

    def forward(self, x):
        return self.Extraction(x)


class Dense_ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = torch.cat((x, out), 1)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Dense_Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.add_module('dense_conv' + str(i),
                            Dense_ConvLayer(in_channels + i * out_channels, out_channels, kernel_size, stride))
        self.adjust_conv = ConvLayer(in_channels + num_layers * out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            # print('num_block - ' + str(i))
            dense_conv = getattr(self, 'dense_conv' + str(i))
            out = dense_conv(out)
        out = self.adjust_conv(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode="reflect"),
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 out_fratures,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias, padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Restormer_Encoder(nn.Module):
    def __init__(self):
        super(Restormer_Encoder, self).__init__()

        # channel = [32, 64, 128, 256]
        channel = [8, 16, 32, 64]
        self.en_1 = nn.Sequential(*[
            Restormer_CNN_block(channel[0], channel[0]),
            nn.InstanceNorm2d(channel[0]),
            nn.ReLU(True),
        ])
        self.en_2 = nn.Sequential(*[
            Restormer_CNN_block(channel[0], channel[0]),
            nn.InstanceNorm2d(channel[0]),
            nn.ReLU(True),
        ])
        self.en_3 = nn.Sequential(*[
            Restormer_CNN_block(channel[1], channel[1]),
            nn.InstanceNorm2d(channel[1]),
            nn.ReLU(True),
        ])
        self.en_4 = nn.Sequential(*[
            Restormer_CNN_block(channel[2], channel[2]),
            nn.InstanceNorm2d(channel[2]),
            nn.ReLU(True),
        ])


        self.first = nn.Sequential(*[
            nn.Conv2d(1, channel[0], kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(channel[0]),
            nn.ReLU(True)
        ])


        self.down1 = nn.Sequential(*[
            nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(channel[0]),
            # nn.ReLU(True),
        ])

        self.down2 = nn.Sequential(*[
            nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1, bias=True,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(channel[1]),
            # nn.ReLU(True),

        ])

        self.down3 = nn.Sequential(*[

            nn.Conv2d(channel[1], channel[2], kernel_size=3, stride=2, padding=1, bias=True,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(channel[2]),
            # nn.ReLU(True),

        ])

        self.down4 = nn.Sequential(*[
            nn.Conv2d(channel[2], channel[3], kernel_size=3, stride=2, padding=1, bias=True,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(channel[3]),
            # nn.ReLU(True),

        ])

    def forward(self, img):
        img0 = self.first(img)
        img1 = self.down1(self.en_1(img0) + img0)
        img2 = self.down2(self.en_2(img1) + img1)
        img3 = self.down3(self.en_3(img2) + img2)
        img4 = self.down4(self.en_4(img3) + img3)

        return [img1, img2, img3, img4]


class Restormer_Decoder(nn.Module):
    def __init__(self):
        super(Restormer_Decoder, self).__init__()
        channel = [8, 16, 32, 64]
        self.de_1 = Restormer_CNN_block(channel[0] * 2, channel[0])
        self.de_2 = Restormer_CNN_block(channel[1] * 2, channel[1])
        self.de_3 = Restormer_CNN_block(channel[2] * 2, channel[2])
        self.de_4 = Restormer_CNN_block(channel[3], channel[3])



        self.up4 = nn.Sequential(
            self.de_4,
            # nn.InstanceNorm2d(channel[3]),
            nn.ConvTranspose2d(channel[3], channel[2], 4, 2, 1, bias=True),
            nn.InstanceNorm2d(channel[2]),
            nn.ReLU(True),

        )
        self.up3 = nn.Sequential(
            self.de_3,
            # nn.InstanceNorm2d(channel[2]),
            nn.ConvTranspose2d(channel[2], channel[1], 4, 2, 1, bias=True),
            nn.InstanceNorm2d(channel[1]),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            self.de_2,
            # nn.InstanceNorm2d(channel[1]),
            nn.ConvTranspose2d(channel[1], channel[0], 4, 2, 1, bias=True),
            nn.InstanceNorm2d(channel[0]),
            nn.ReLU(True),

        )

        self.de_1_last = nn.Sequential(
            self.de_1,
            # nn.InstanceNorm2d(channel[0]),
            nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(channel[0]),
            nn.ReLU(True),
        )


        self.last = nn.Sequential(
            nn.Conv2d(channel[0], 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            # nn.InstanceNorm2d(channel[0]),
            nn.Tanh()
        )

    def forward(self,x):


        out=self.up4(x[3])
        out=self.up3(torch.cat((out,x[2]),1))
        out=self.up2(torch.cat((out,x[1]),1))
        out=self.de_1_last(torch.cat((out,x[0]),1))
        out = self.last(out)
        return out


EPSILON = 1e-8
class Weight(nn.Module):
    def __init__(self, ch_1, ch_2, ch_3, ch_4):
        super(Weight, self).__init__()
        # weight for features
        # ks_s = 17
        ks_s = 3
        ks_d = 3
        weight_1 = torch.ones([1, 1, ks_s, ks_s])
        reflection_padding_sh = int(np.floor(ks_s / 2))
        # weight_sh = torch.from_numpy(np.array([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]))
        # ch_s_2 = int(ch_s/2)
        self.conv_1 = torch.nn.Conv2d(ch_1, ch_1, (ks_s, ks_s), stride=1, padding=reflection_padding_sh, bias=False)
        self.conv_1.weight.data = (1 / (ks_s * ks_s)) * weight_1.repeat(ch_1, ch_1, 1, 1).float()
        self.conv_1.requires_grad_(False)

        weight_2 = torch.ones([1, 1, ks_d, ks_d])
        reflection_padding_de = int(np.floor(ks_d / 2))
        self.conv_2 = torch.nn.Conv2d(ch_2, ch_2, (ks_d, ks_d), stride=1, padding=reflection_padding_de, bias=False)
        self.conv_2.weight.data = (1 / (ks_d * ks_d)) * weight_2.repeat(ch_2, ch_2, 1, 1).float()
        self.conv_2.requires_grad_(False)

        weight_3 = torch.ones([1, 1, ks_d, ks_d])
        reflection_padding_de = int(np.floor(ks_d / 2))
        self.conv_3 = torch.nn.Conv2d(ch_3, ch_3, (ks_d, ks_d), stride=1, padding=reflection_padding_de, bias=False)
        self.conv_3.weight.data = (1 / (ks_d * ks_d)) * weight_3.repeat(ch_3, ch_3, 1, 1).float()
        self.conv_3.requires_grad_(False)

        weight_4 = torch.ones([1, 1, ks_d, ks_d])
        reflection_padding_de = int(np.floor(ks_d / 2))
        self.conv_4 = torch.nn.Conv2d(ch_4, ch_4, (ks_d, ks_d), stride=1, padding=reflection_padding_de, bias=False)
        self.conv_4.weight.data = (1 / (ks_d * ks_d)) * weight_4.repeat(ch_4, ch_4, 1, 1).float()
        self.conv_4.requires_grad_(False)



    def for_1(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = x - self.conv_sh(x)
        # g_y = y - self.conv_sh(y)
        g_x = torch.sqrt((x - self.conv_1(x)) ** 2)
        g_y = torch.sqrt((y - self.conv_1(y)) ** 2)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)

        w_x = w_x.detach()
        w_y = w_y.detach()

        return w_x, w_y

    def for_2(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = torch.sqrt((x - self.conv_de(x)) ** 2)
        # g_y = torch.sqrt((y - self.conv_de(y)) ** 2)
        g_x = torch.sqrt((x - self.conv_2(x)) ** 2)
        g_y = torch.sqrt((y - self.conv_2(y)) ** 2)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)

        w_x = w_x.detach()
        w_y = w_y.detach()
        return w_x, w_y


    def for_3(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        # g_x = torch.sqrt((x - self.conv_de(x)) ** 2)
        # g_y = torch.sqrt((y - self.conv_de(y)) ** 2)
        g_x = self.conv_3(x)
        g_y = self.conv_3(y)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)

        w_x = w_x.detach()
        w_y = w_y.detach()
        return w_x, w_y


    def for_4(self, x, y):
        channels1 = x.size()[1]
        channels2 = y.size()[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."

        g_x = self.conv_4(x)
        g_y = self.conv_4(y)
        w_x = g_x / (g_x + g_y + EPSILON)
        w_y = g_y / (g_x + g_y + EPSILON)

        w_x = w_x.detach()
        w_y = w_y.detach()
        return w_x, w_y

class Fusion_Decoder(nn.Module):
    def __init__(self):
        super(Fusion_Decoder, self).__init__()
        channel_generate = [8, 16, 32, 64]
        channel_extract = [8, 16, 32, 64]

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(channel_extract[3], channel_extract[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(channel_generate[2])),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channel_extract[2], channel_extract[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(channel_generate[1])),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channel_extract[1], channel_extract[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(channel_generate[0])),
        )

        self.vmd_I = vmd(1)
        self.vmd_V = vmd(1)

        self.res_I = nn.Sequential(*[ResBlock(channel_extract[3],channel_extract[3]) for i in range(3)])

        self.res_V = nn.Sequential(*[ResBlock(channel_extract[3],channel_extract[3]) for i in range(3)])


        self.de_1 = Restormer_CNN_block(channel_extract[0] * 2, channel_extract[0])
        self.de_2 = Restormer_CNN_block(channel_extract[1] * 2, channel_extract[1])
        self.de_3 = Restormer_CNN_block(channel_extract[2] * 2, channel_extract[2])
        self.de_4 = Restormer_CNN_block(channel_extract[3], channel_extract[3])

        self.weight = Weight(channel_extract[0],channel_extract[1],channel_extract[2],channel_extract[3])



        self.last = nn.Sequential(
            nn.Conv2d(channel_extract[0], 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.Tanh()
        )





        self.f_1 = nn.Sequential(*[Restormer_CNN_block(channel_extract[0] * 2, channel_extract[0]),
                                     nn.BatchNorm2d(channel_extract[0]),
                                   ]
                                   )
        self.f_2 = nn.Sequential(*[Restormer_CNN_block(channel_extract[1] * 2, channel_extract[1]),
                                     nn.BatchNorm2d(channel_extract[1]),
                                   ]
                                   )
        self.f_3 = nn.Sequential(*[Restormer_CNN_block(channel_extract[2] * 2, channel_extract[2]),
                                     nn.BatchNorm2d(channel_extract[2]),
                                   ]
                                   )
        self.f_4 = nn.Sequential(*[Restormer_CNN_block(channel_extract[3] * 2, channel_extract[3]),
                                     nn.BatchNorm2d(channel_extract[3]),
                                   ]
                                   )

        # 此处可以用 batchnorm

    def forward(self, I_generate, I_extract, V_generate, V_extract):
        # I_generate  V_extract   detail
        # I_extract   V_generate  light


        I_extract, I_loss = self.vmd_I(I_generate,I_extract)
        V_extract, V_loss = self.vmd_V(V_generate,V_extract)


        f_4 = self.f_4(torch.cat((V_extract[3],I_extract[3]),dim=1))
        f_3 = self.f_3(torch.cat((V_extract[2],I_extract[2]),dim=1))
        f_2 = self.f_2(torch.cat((V_extract[1],I_extract[1]),dim=1))
        f_1 = self.f_1(torch.cat((V_extract[0],I_extract[0]),dim=1))

        out = self.up4(self.de_4(f_4))
        out = self.up3(self.de_3(torch.cat((out, f_3), 1)))
        out = self.up2(self.de_2(torch.cat((out, f_2), 1)))
        out = self.de_1(torch.cat((out, f_1), 1))
        out = self.last(out)



        return out



