from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

import log
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


class CTDenoiser(object):
	"""Implementation of Noise2Noise from Lehtinen et al. (2018)."""

	def __init__(self, loss, noise_type, learning_rate, adam, nb_epochs, cuda, trainable):
		"""Initializes model."""

		self.trainable = trainable
		self._noise_type = noise_type
		self._nb_epochs = nb_epochs
		self._use_cuda = cuda and torch.cuda.is_available()
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

	def plot_per_epoch(self, ckpt_dir, title, measurements, y_label):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(range(1, len(measurements) + 1), measurements)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_xlabel('Epoch')
		ax.set_ylabel(y_label)
		ax.set_title(title)
		plt.tight_layout()

		plt.savefig(Path(ckpt_dir, f"{title.replace(' ', '-').lower()}.png"), dpi=200)
		plt.close()

	def save_model(self, epoch, stats, ckpt_dir, ckpt_overwrite, first=False):
		"""Saves model to files; can be overwritten at every epoch to save disk space."""
		# Save checkpoint dictionary
		if ckpt_overwrite:
			fname_unet = f'{ckpt_dir}/n2n-{self._noise_type}.pt'
		else:
			valid_loss = stats['valid_loss'][epoch]
			fname_unet = f'{ckpt_dir}/n2n-epoch{epoch + 1}-{valid_loss:>1.5f}.pt'

		log.log('Model Saving', f'Checkpoint {fname_unet}')
		torch.save(self.model.state_dict(), fname_unet)

		# Save stats to JSON
		with open(f'{ckpt_dir}/n2n-stats.json', 'w') as fp:
			json.dump(stats, fp, indent=2)

	def time_elapsed_since(self, start):
		timedelta = datetime.now() - start
		string = str(timedelta)[:-7]
		ms = int(timedelta.total_seconds() * 1000)
		return string, ms

	def show_on_epoch_end(self, epoch_time, valid_time, valid_loss, valid_psnr):
		log.log("Epoch Completion",
			f'Train time: {epoch_time} ; Valid time: {valid_time} ; '
			f'Valid loss: {valid_loss:>1.5f} ; Avg PSNR: {valid_psnr:.2f} dB')

	def load_model(self, ckpt_fname):
		"""Loads model from checkpoint file."""
		log.log('Loading Checkpoint', ckpt_fname)
		if self.use_cuda:
			self.model.load_state_dict(torch.load(ckpt_fname))
		else:
			self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

	def progress_bar(self, batch_idx, num_batches, report_interval, train_loss):
		dec = int(np.ceil(np.log10(num_batches)))
		bar_size = 21 + dec
		progress = (batch_idx % report_interval) / report_interval
		fill = int(progress * bar_size) + 1
		print(f'\rBatch {batch_idx + 1:>{dec}d} [{"=" * fill}>{" " * (bar_size - fill)}] Train loss: {train_loss:>1.5f}')

	def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader,
						plot_stats, ckpt_dir, ckpt_overwrite):
		# Evaluate model on validation set
		epoch_time = self.time_elapsed_since(epoch_start)[0]
		valid_loss, valid_time, valid_psnr = self.model.eval(valid_loader)
		self.show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

		# Decrease learning rate if plateau
		self.scheduler.step(valid_loss)

		# Save checkpoint
		stats['train_loss'].append(train_loss)
		stats['valid_loss'].append(valid_loss)
		stats['valid_psnr'].append(valid_psnr)
		self.save_model(epoch, stats, ckpt_dir, ckpt_overwrite, epoch == 0)

		# Plot stats
		if plot_stats:
			self.plot_per_epoch(ckpt_dir, 'Valid loss', stats['valid_loss'], f'{self._loss_str} Loss')
			self.plot_per_epoch(ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

	def __create_checkpoint_dir(self, ckpt_save_path, ckpt_overwrite):
		# Create directory for model checkpoints, if nonexistent
		if ckpt_overwrite:
			ckpt_dir_name = self._noise_type
		else:
			ckpt_dir_name = f'{datetime.now():{self._noise_type}-covidct-%H%M}'

		ckpt_dir = Path(ckpt_save_path, ckpt_dir_name)
		ckpt_dir.mkdir(exist_ok=True, parents=True)
		return ckpt_dir

	def train(self, train_loader, valid_loader, report_interval, plot_stats, ckpt_save_path, ckpt_overwrite):
		"""Trains denoiser on training set."""
		self.model.train(True)

		# self._print_params()
		num_batches = len(train_loader)
		assert num_batches % report_interval == 0, 'Report interval must divide total number of batches'

		# Dictionaries of tracked stats
		stats = {'noise_type': 'natural',
				'noise_param': 'nextslice',
				'train_loss': [],
				'valid_loss': [],
				'valid_psnr': []}

		# Main training loop
		train_start = datetime.now()

		ckpt_dir = self.__create_checkpoint_dir(ckpt_save_path, ckpt_overwrite)

		for epoch in range(self._nb_epochs):
			log.log('Training Epoch', f'EPOCH {epoch + 1:d} / {self._nb_epochs:d}')

			# Some stats trackers
			epoch_start = datetime.now()
			train_loss_meter = AvgMeter()
			loss_meter = AvgMeter()
			time_meter = AvgMeter()

			# Minibatch SGD
			for batch_idx, (source, target) in enumerate(train_loader):
				batch_start = datetime.now()
				self.progress_bar(batch_idx, num_batches, report_interval, loss_meter.val)

				if self._use_cuda:
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
				if (batch_idx + 1) % report_interval == 0 and batch_idx:
					show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
					train_loss_meter.update(loss_meter.avg)
					loss_meter.reset()
					time_meter.reset()

			# Epoch end, save and reset tracker
			self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader,
								plot_stats, ckpt_dir, ckpt_overwrite)
			train_loss_meter.reset()

		train_elapsed = self.time_elapsed_since(train_start)[0]
		log.log('Training Complete', f'Elapsed Time {train_elapsed}')


def show_on_report(batch_idx, num_batches, loss, elapsed):
	dec = int(np.ceil(np.log10(num_batches)))
	log.log('Batch Complete',
		f'{batch_idx + 1:>{dec}d} / {num_batches:d} | Avg loss: {loss:>1.5f} | Avg train time / batch: {int(elapsed):d} ms')
