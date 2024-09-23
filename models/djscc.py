#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os, sys
os.chdir(sys.path[0])
sys.path.append(os.getcwd())
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .channel import Channel
from fractions import Fraction


class EnergyNormalizationLayer(nn.Module):
    def __init__(self):
        super(EnergyNormalizationLayer, self).__init__()

    def forward(self, x):
        # 计算每个特征图的能量（L2范数）
        energy = torch.sqrt(torch.sum(x ** 2, dim=2, keepdim=True)) + 1e-8  # 添加一个小的常数避免除零错误
        # 归一化每个特征图的能量
        normalized_x = x / energy
        return normalized_x


class DjsccEncoder(nn.Module):
    def __init__(self,
                 k,
                 c_list = [64, 256, 1024],
                 in_chans=3,
                 **kwargs):

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, c_list[0], 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(c_list[0], c_list[1], 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(c_list[1], c_list[2], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(c_list[2], 32, 3, 1, 1),
            nn.ReLU())
        self.energy_norm_layer = EnergyNormalizationLayer()
        self.liner = nn.Linear(32, k)


    def forward(self, x):
        x = self.net(x)
        x_h = x.shape[2]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.liner(x)
        x = self.energy_norm_layer(x)
        return x, x_h

        

class DjsccDecoder(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 k,
                 c_list = [64, 256, 1024],
                 **kwargs):
        super().__init__()
        self.liner = nn.Linear(k, 32)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, c_list[2], 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_list[2], c_list[1], 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_list[1], c_list[0], 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_list[0], 3, 3, 2, 1, output_padding=1),
            nn.Sigmoid())
    

    def forward(self, x, x_h):
        x = self.liner(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=x_h)
        x = nn.ReLU()(x)
        x = self.net(x)
        return x


class Djscc(nn.Module):

    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.encoder = DjsccEncoder(8,[16,32,32])
        self.decoder = DjsccDecoder(8,[16,32,32])
        self.channel = Channel(config)
        self.multiple_snr = config.multiple_snr


    def forward(self, x, given_snr=False):
        semantic_feature, x_h = self.encoder(x)
        CBR = Fraction(semantic_feature.numel()) / 2 / Fraction(x.numel())
        if given_snr:
            g_snr = given_snr
            choice = self.multiple_snr.index(g_snr)
        else:
            choice = random.randint(0, len(self.multiple_snr) - 1)
            g_snr = self.multiple_snr[choice]
        x_signal = self.channel(semantic_feature, g_snr)

        snr = 10*torch.log10(torch.mean(semantic_feature**2, dim=[1, 2]) / torch.mean((x_signal-semantic_feature)**2, dim=[1, 2]))

        x = self.decoder(x_signal, x_h)
        cla = torch.rand((x_signal.shape[0], 10)).cuda()
        return x, CBR, g_snr, snr, cla, semantic_feature, x_signal
