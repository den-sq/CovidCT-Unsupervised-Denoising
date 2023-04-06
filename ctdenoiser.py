from datetime import datetime
import os
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from network import UNet


class AvgMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0.
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class CTdenoiser(object):
	"""Implementation of Noise2Noise from Lehtinen et al. (2018)."""

	def __init__(self, params, trainable):
		"""Initializes model."""

		self.p = params
		self.trainable = trainable
		self._compile()
		self._use_cuda = self.p.cuda and torch.cuda.is_available()

	def _compile(self):
		"""Compiles model (architecture, loss function, optimizers, etc.)."""

		self.model = UNet(in_channels=1)

		# Set optimizer and loss, if in training mode
		if self.trainable:
			self.optim = Adam(self.model.parameters(), lr=self.p.learning_rate, betas=self.p.adam[:2], Seps=self.p.adam[2])

			# Learning rate adjustment
			self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, patience=self.p.nb_epochs / 4,
															factor=0.5, verbose=True)

			# Loss function
			if self.p.loss == 'l2':
				self.loss = nn.MSELoss()
			else:
				self.loss = nn.L1Loss()
			if self._use_cuda:
				self.loss = self.loss.cuda()

		# CUDA support
		if self._use_cuda:
			self.model = self.model.cuda()

	def _print_params(self):
		"""Formats parameters to print when training."""
		param_dict = vars(self.p)

		def pretty(x):
			return x.replace('_', ' ').capitalize()

		print('Training parameters: ')
		print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
		print()

	def plot_per_epoch(self, ckpt_dir, title, measurements, y_label):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(range(1, len(measurements) + 1), measurements)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_xlabel('Epoch')
		ax.set_ylabel(y_label)
		ax.set_title(title)
		plt.tight_layout()

		fname = '{}.png'.format(title.replace(' ', '-').lower())
		plot_fname = os.path.join(ckpt_dir, fname)
		plt.savefig(plot_fname, dpi=200)
		plt.close()

	def save_model(self, epoch, stats, first=False):
		"""Saves model to files; can be overwritten at every epoch to save disk space."""

		# Create directory for model checkpoints, if nonexistent
		if first:
			ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-covidct-%H%M}'

			if self.p.ckpt_overwrite:
				ckpt_dir_name = self.p.noise_type

			self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
			if not os.path.isdir(self.p.ckpt_save_path):
				os.mkdir(self.p.ckpt_save_path)
			if not os.path.isdir(self.ckpt_dir):
				os.mkdir(self.ckpt_dir)

		# Save checkpoint dictionary
		if self.p.ckpt_overwrite:
			fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)
		else:
			valid_loss = stats['valid_loss'][epoch]
			fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
		print('Saving checkpoint to: {}\n'.format(fname_unet))
		torch.save(self.model.state_dict(), fname_unet)

		# Save stats to JSON
		fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
		with open(fname_dict, 'w') as fp:
			json.dump(stats, fp, indent=2)

	def time_elapsed_since(self, start):
		timedelta = datetime.now() - start
		string = str(timedelta)[:-7]
		ms = int(timedelta.total_seconds() * 1000)
		return string, ms

	def show_on_epoch_end(self, epoch_time, valid_time, valid_loss, valid_psnr):
		print('\r{}'.format(' ' * 80), end='\r')
		print(f'Train time: {epoch_time} | Valid time: {valid_time} |'
				f' Valid loss: {valid_loss:>1.5f} | Avg PSNR: {valid_psnr:.2f} dB')

	def load_model(self, ckpt_fname):
		"""Loads model from checkpoint file."""
		print('Loading checkpoint from: {}'.format(ckpt_fname))
		if self.use_cuda:
			self.model.load_state_dict(torch.load(ckpt_fname))
		else:
			self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

	def progress_bar(self, batch_idx, num_batches, report_interval, train_loss):
		dec = int(np.ceil(np.log10(num_batches)))
		bar_size = 21 + dec
		progress = (batch_idx % report_interval) / report_interval
		fill = int(progress * bar_size) + 1
		print(f'\rBatch {batch_idx + 1:>{dec}d} [{"=" * fill}>{" " * (bar_size - fill)}]'
				f' Train loss: {train_loss:>1.5f}'.format(dec=str(dec)))

	def show_on_report(batch_idx, num_batches, loss, elapsed):
		print('\r{}'.format(' ' * 80), end='\r')
		dec = int(np.ceil(np.log10(num_batches)))
		print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(
				batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))

	def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
		# Evaluate model on validation set
		print('\rTesting model on validation set... ', end='')
		epoch_time = self.time_elapsed_since(epoch_start)[0]
		valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
		self.show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

		# Decrease learning rate if plateau
		self.scheduler.step(valid_loss)

		# Save checkpoint
		stats['train_loss'].append(train_loss)
		stats['valid_loss'].append(valid_loss)
		stats['valid_psnr'].append(valid_psnr)
		self.save_model(epoch, stats, epoch == 0)

		# Plot stats
		if self.p.plot_stats:
			loss_str = f'{self.p.loss.upper()} loss'
			self.plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
			self.plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

	def train(self, train_loader, valid_loader):
		"""Trains denoiser on training set."""
		self.model.train(True)

		self._print_params()
		num_batches = len(train_loader)
		assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

		# Dictionaries of tracked stats
		stats = {'noise_type': 'natural',
				'noise_param': 'nextslice',
				'train_loss': [],
				'valid_loss': [],
				'valid_psnr': []}

		# Main training loop
		train_start = datetime.now()
		for epoch in range(self.p.nb_epochs):
			print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

			# Some stats trackers
			epoch_start = datetime.now()
			train_loss_meter = AvgMeter()
			loss_meter = AvgMeter()
			time_meter = AvgMeter()

			# Minibatch SGD
			for batch_idx, (source, target) in enumerate(train_loader):
				batch_start = datetime.now()
				self.progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

				if self.p.cuda:
					source = source.cuda()
					target = target.cuda()

				# Denoise image
				source_denoised = self.model(source)

				loss = self.loss(source_denoised, target)
				loss_meter.update(loss.item())

				# Zero gradients, perform a backward pass, and update the weights
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()

				# Report/update statistics
				time_meter.update(self.time_elapsed_since(batch_start)[1])
				if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
					self.show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
					train_loss_meter.update(loss_meter.avg)
					loss_meter.reset()
					time_meter.reset()

			# Epoch end, save and reset tracker
			self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
			train_loss_meter.reset()

		train_elapsed = self.time_elapsed_since(train_start)[0]
		print('Training done! Total elapsed time: {}\n'.format(train_elapsed))
