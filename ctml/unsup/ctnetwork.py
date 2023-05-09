#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

use_cuda = False
nodes = 1

class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=1, out_channels=1, mid_layers=4):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        self._encoders = nn.ModuleDict({
                # Layers: enc_conv0, enc_conv1, pool1
                "convulute_start": nn.Sequential(
                    nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(48, 48, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))
            } | {
                # Layers: enc_conv(i), pool(i); i=2..n + 1
                f"convolute_{x + 1}": nn.Sequential(
                    nn.Conv2d(48, 48, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))
                for x in range(0, mid_layers)
            })

        # Layers: enc_conv(n + 2), upsample(n + 1)
        self._encoder_convolute_to_upsample = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=0))

        self._decoders = nn.ModuleDict({
                # Layers: dec_conv5a, dec_conv5b, upsample4
                "deconvolute_start": nn.Sequential(
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0))
            } | {
                # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=n..2
                f"deconvolute_{x}": nn.Sequential(
                    nn.Conv2d(144, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 96, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0))
                for x in range(mid_layers, 1, -1)
            } | {
                # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
                "deconvolute_end": nn.Sequential(
                    nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.1))
            })

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool = [x]
        for _id, enc in self._encoders.items():
            pool.append(enc(pool[-1]))

        # Start upsampling
        upsample = self._encoder_convolute_to_upsample(pool[-1])

        # Decoding and final activatation.
        for i, (_id, dec) in enumerate(self._decoders.items()):
            concat = torch.cat((upsample, pool[-(i + 2)]), dim=1)
            upsample = dec(concat)

        # Return finally activated sample
        return upsample
