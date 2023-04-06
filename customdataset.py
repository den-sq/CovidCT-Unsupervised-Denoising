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

	def _norma(self, img):
		dec = (np.max(img) - np.min(img))
		if dec == 0.0:
			dec = 0.00001
		self.norm = (img - np.min(img)) / dec
		return self.norm

	def __len__(self):
		"""Returns length of dataset."""
		return self.__data_size

	def __getitem__(self, index):
		"""Retrieves images from folder and creates iterator"""

		# Load images
		inputimage = self.inputs[index]
		targetimage = self.targets[index]
		inpimg = tf.imread(inputimage)
		tarimg = tf.imread(targetimage)
		inpimg, tarimg = self._random_crop(inpimg, tarimg)

		inpimg = self.trans(self._norma(inpimg))
		tarimg = self.trans(self._norma(tarimg))

		return inpimg, tarimg
