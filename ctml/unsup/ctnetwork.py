#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

use_cuda = False


class UNet(nn.Module):
	"""Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

	def __init__(self, in_channels=1, out_channels=1):
		"""Initializes U-Net."""

		super(UNet, self).__init__()

		# Layers: enc_conv0, enc_conv1, pool1
		self._block1 = nn.Sequential(
			nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(48, 48, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2))

		# Layers: enc_conv(i), pool(i); i=2..5
		self._block2, self._block2a, self._block2b, self._block2c = [nn.Sequential(
			nn.Conv2d(48, 48, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2)) for _ in range(4)]

		# Layers: enc_conv6, upsample5
		self._block3 = nn.Sequential(
			nn.Conv2d(48, 48, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ReflectionPad2d(1),
			nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=0))

		# Layers: dec_conv5a, dec_conv5b, upsample4
		self._block4 = nn.Sequential(
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ReflectionPad2d(1),
			nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0))

		# Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
		self._block5, self._block5a, self._block5b = [nn.Sequential(
			nn.Conv2d(144, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ReflectionPad2d(1),
			nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0)) for _ in range(3)]

		# Layers: dec_conv1a, dec_conv1b, dec_conv1c,
		self._block6 = nn.Sequential(
			nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
			nn.LeakyReLU(0.1))

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
		pool1 = self._block1(x)
		pool2 = self._block2(pool1)
		pool3 = self._block2a(pool2)
		pool4 = self._block2b(pool3)
		pool5 = self._block2c(pool4)

		# Decoder
		upsample = self._block3(pool5)
		concat = torch.cat((upsample, pool4), dim=1)
		upsample = self._block4(concat)
		concat = torch.cat((upsample, pool3), dim=1)
		upsample = self._block5(concat)
		concat = torch.cat((upsample, pool2), dim=1)
		upsample = self._block5a(concat)
		concat = torch.cat((upsample, pool1), dim=1)
		upsample = self._block5b(concat)
		concat = torch.cat((upsample, x), dim=1)

		# Final activation
		return self._block6(concat)
