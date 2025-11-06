# -*- coding: utf-8 -*- 
# @Time : 2024/10/24 15:56 
# @Author : bfliu Beijing University of Posts and Telecommunications
# @File : generate_model.py
# @contact: bfliu@bupt.edu.cn

import functools

import torch
import torch.nn as nn




class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def  __init__(self, input_nc, output_nc, ngf=64, norm_layer= functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False), use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self,input):
        aa = None
        for i,layer in enumerate(self.model):
            input = layer(input)
            # if i == 15:
            #     aa = input
            #     print(layer)
            #     break
                # return input
        # return aa
        return input



class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator_Encoder_decoder(nn.Module):
    def __init__(self):
        super(ResnetGenerator_Encoder_decoder, self).__init__()

        self.ResnetGenerator_Encoder = ResnetGenerator_Encoder(1,1)
        self.ResnetGenerator_Decoder = ResnetGenerator_Decoder(1,1)

    def forward(self, input):
        out = self.ResnetGenerator_Encoder(input)
        # out = self.ResnetGenerator_Decoder(out)
        return out




class ResnetGenerator_Encoder(nn.Module):

    def  __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_Encoder, self).__init__()
        use_bias = True

        channels = [8,16,32,64]


        self.fe1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, channels[0], kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(channels[0]),
                 nn.ReLU(True)
                 ])

        self.fe2 = nn.Sequential(*[nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(channels[1]),
                      nn.ReLU(True)])

        self.fe3 = nn.Sequential(*[nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(channels[2]),
                    nn.ReLU(True)])

        self.fe4 = nn.Sequential(*[nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(channels[3]),
                    nn.ReLU(True)])

        # self.fe3_de3 = nn.Sequential(*[
        #     ResnetBlock(channels[2], padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
        #                 use_bias=use_bias)
        #     for i in range(9)
        # ])

        self.fe4_de4 = nn.Sequential(*[
            ResnetBlock(channels[3], padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias)
            for i in range(9)
        ])

    def forward(self, input):
        """Standard forward"""
        # input /= 255.0
        img1 = self.fe1(input)
        # print(img1.size())
        img2 = self.fe2(img1)
        # print(img2.size())
        #
        img3 = self.fe3(img2)
        # print(img3.size())

        img4 = self.fe4(img3)

        # img3 = self.fe3_de3(img3)

        img4 = self.fe4_de4(img4)

        return [img1,img2,img3,img4]


class ResnetGenerator_Decoder(nn.Module):

    def  __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator_Decoder, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        channels = [8,16,32,64]

        self.de4 = nn.Sequential(*[nn.ConvTranspose2d(channels[3], channels[2],
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(channels[3]),
                      nn.ReLU(True)])
        self.de3 = nn.Sequential(*[nn.ConvTranspose2d(channels[2], channels[1],
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(channels[3]),
                      nn.ReLU(True)])
        self.de2 = nn.Sequential(*[nn.ConvTranspose2d(channels[1], channels[0],
                                                      kernel_size=3, stride=2,
                                                      padding=1, output_padding=1,
                                                      bias=use_bias),
                                   norm_layer(channels[3]),
                                   nn.ReLU(True)])

        self.de1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                 nn.Conv2d(channels[0], 1, kernel_size=7, padding=0, bias=use_bias),
                 # norm_layer(channels[0]),
                 nn.Tanh(),
                 ])




    def forward(self, input):
        img_out = self.de4(input[3])
        img_out = self.de3(img_out)
        img_out = self.de2(img_out)
        img_out = self.de1(img_out)

        return img_out
