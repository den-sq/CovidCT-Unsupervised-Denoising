from pathlib import Path
import random

from natsort import natsorted
import numpy as np
import tifffile as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CTDataset(Dataset):
	def __init__(self, root_dir, crop_size=512):
		"""Initializes abstract dataset."""
		super(CTDataset, self).__init__()
		self.imgs = []
		self.root_dir = Path(root_dir)
		self.crop_size = crop_size

		if root_dir.exists():
			self.inputs = natsorted(root_dir.iterdir())
			self.targets = self.inputs[1:] + self.inputs[-2]
			self.__data_size = len(self.inputs)
			self.trans = transforms.Compose([transforms.ToTensor()])

	def loader(self, batch_size, single=False, shuffle=False):
		return DataLoader(self, batch_size=1 if single else batch_size, shuffle=shuffle)

	def _random_crop(self, image1, image2):
		m, n = image1.shape
		randx = random.randint(0, m - (self.crop_size + 5))
		randy = random.randint(0, n - (self.crop_size + 5))
		image1 = image1[randx:randx + self.crop_size, randy:randy + self.crop_size]
		image2 = image2[randx:randx + self.crop_size, randy:randy + self.crop_size]
		return image1, image2

	def _norma(self, img, bottom_threshold, top_threshold):
		""" Basic Image Normalization. """
		floor = np.percentile(img, bottom_threshold)
		ceiling = np.percentile(img, top_threshold)

		normalized = img - floor
		normalized[normalized < 0] = 0

		return normalized / (ceiling - floor)

	def __len__(self):
		"""Returns length of dataset."""
		return self.__data_size

	def __getitem__(self, index, normalize_over):
		"""Retrieves images from folder and creates iterator"""

		# Load images
		inputimage = self.inputs[index]
		targetimage = self.targets[index]
		inpimg = tf.imread(inputimage)
		tarimg = tf.imread(targetimage)
		inpimg, tarimg = self._random_crop(inpimg, tarimg)

		if normalize_over is not None:
			return (self.trans(self._norma(inpimg, normalize_over.start, normalize_over.stop)),
					self.trans(self._norma(tarimg, normalize_over.start, normalize_over.stop)))
		else:
			return self.trans(inpimg), self.trans(tarimg)
