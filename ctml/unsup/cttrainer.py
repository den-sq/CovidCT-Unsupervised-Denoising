from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from util import log
from unsup.ctnetwork import UNet


class CTTrainer(object):
	"""Implementation of Noise2Noise from Lehtinen et al. (2018)."""

	def __init__(self, loss, noise_type, learning_rate, adam, nb_epochs, cuda, trainable):
		"""Initializes model."""

		self.trainable = trainable
		self._noise_type = noise_type
		self._nb_epochs = nb_epochs
		self._use_cuda = cuda
		self._loss_str = loss

		self._compile(loss, learning_rate, adam)

	def _compile(self, loss, learning_rate, adam):
		"""Compiles model (architecture, loss function, optimizers, etc.)."""

		self.model = UNet(in_channels=1)

		# Set optimizer and loss, if in training mode
		if self.trainable:
			self.optim = Adam(self.model.parameters(), lr=learning_rate, betas=adam[:2], eps=adam[2])

			# Learning rate adjustment
			self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, patience=self._nb_epochs / 4,
															factor=0.5, verbose=True)

			# Loss function
			if loss == 'l2':
				self.loss = nn.MSELoss()
			else:
				self.loss = nn.L1Loss()
			if self._use_cuda:
				self.loss = self.loss.cuda()

		# CUDA support
		if self._use_cuda:
			self.model = self.model.cuda()

	def save_model(self, epoch, validation_loss, ckpt_dir, ckpt_overwrite, first=False):
		"""Saves model to files; can be overwritten at every epoch to save disk space."""
		# Save checkpoint dictionary
		if ckpt_overwrite:
			fname_unet = f'{ckpt_dir}/n2n-{self._noise_type}.pt'
		else:
			fname_unet = f'{ckpt_dir}/n2n-epoch{epoch + 1}-{validation_loss:>1.5f}.pt'

		log.log('Model Saving', f'Checkpoint {fname_unet}')
		torch.save(self.model.state_dict(), fname_unet)

	def load_model(self, ckpt_fname):
		"""Loads model from checkpoint file."""
		log.log('Loading Checkpoint', ckpt_fname)
		if self._use_cuda:
			self.model.load_state_dict(torch.load(ckpt_fname))
		else:
			self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

	def __create_checkpoint_dir(self, ckpt_save_path, ckpt_overwrite):
		# Create directory for model checkpoints, if nonexistent
		if ckpt_overwrite:
			ckpt_dir_name = self._noise_type
		else:
			ckpt_dir_name = f'{datetime.now():{self._noise_type}-covidct-%H%M}'

		ckpt_dir = Path(ckpt_save_path, ckpt_dir_name)
		ckpt_dir.mkdir(exist_ok=True, parents=True)
		return ckpt_dir

	def validate(self, valid_loader):
		"""Evaluates denoiser on validation set."""

		self.model.train(False)

		loss_list = []

		with log.log_progress('Validating Epoch', valid_loader,
				disp=lambda x: '' if not len(loss_list) else f'Loss {loss_list[-1]}') as validation_bar:
			for source, target in validation_bar:
				if self._use_cuda:
					source = source.cuda()
					target = target.cuda()

				loss = self.loss(self.model(source), target)
				loss_list.append(loss.item())

		validation_loss = np.mean(loss_list)

		# Decrease learning rate if plateau
		self.scheduler.step(validation_loss)
		return validation_loss

	def train(self, train_loader, valid_loader, report_interval, plot_stats, ckpt_save_path, ckpt_overwrite):
		"""Trains denoiser on training set."""
		self.model.train(True)

		# report_interval does nothing for now

		# Dictionaries of tracked stats
		stats = {'noise_type': 'natural',
				'noise_param': 'nextslice',
				'train_loss': [],
				'valid_loss': []}

		ckpt_dir = self.__create_checkpoint_dir(ckpt_save_path, ckpt_overwrite)

		for epoch in range(self._nb_epochs):
			log.log('Training Start', f'EPOCH {epoch + 1:d} / {self._nb_epochs:d}')

			# Loss Tracker
			loss_list = []

			with log.log_progress('Training Epoch', train_loader,
					disp=lambda x: '' if not len(loss_list) else f'Loss {loss_list[-1]}') as train_bar:
				for source, target in train_bar:
					if self._use_cuda:
						source = source.cuda()
						target = target.cuda()

					loss = self.loss(self.model(source), target)
					loss_list.append(loss.item())

					# Zero gradients, perform a backward pass, and update the weights
					self.optim.zero_grad()
					loss.backward()
					self.optim.step()

			stats['train_loss'].append(np.mean(loss_list))

			log.log("Epoch Training", f'Training Loss: {stats["train_loss"][-1]:>1.5f}')

			# Evaluate model on validation set
			stats['valid_loss'].append(self.validate(valid_loader))

			log.log("Epoch Validation", f'Validation loss: {stats["valid_loss"][-1]:>1.5f}')

			self.save_model(epoch, stats['valid_loss'][-1], ckpt_dir, ckpt_overwrite, epoch == 0)

			# Save stats to JSON
			with open(f'{ckpt_dir}/n2n-stats.json', 'w') as fp:
				json.dump(stats, fp, indent=2)

		log.log('Training Complete')
