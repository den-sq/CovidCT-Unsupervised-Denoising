import os
import random

from natsort import natsorted
import numpy as np
import tifffile as tl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_dataset(root_dir, params, shuffled=False, single=False):
	"""Loads dataset and returns corresponding data loader."""

	# Instantiate appropriate dataset class
	dataset = Customdata(root_dir, params.crop_size)

	# Use batch size of 1, if requested (e.g. test set)
	if single:
		return DataLoader(dataset, batch_size=1, shuffle=shuffled)
	else:
		return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class Customdata(Dataset):
	def __init__(self, root_dir, crop_size=512):
		"""Initializes abstract dataset."""
		super(Customdata, self).__init__()
		self.imgs = []
		self.root_dir = root_dir
		self.crop_size = crop_size

		assert os.path.exists(root_dir)
		inputpaths = [root_dir + x for x in natsorted(os.listdir(root_dir))]
		targetpaths = []
		for i in range(0, len(inputpaths)):
			try:
				targetpaths.append(inputpaths[i + 1])
			except:
				targetpaths.append(inputpaths[i - 1])
		assert len(targetpaths) == len(inputpaths)
		self.data_size = len(targetpaths)
		self.inputs = inputpaths
		self.targets = targetpaths
		self.trans = transforms.Compose([transforms.ToTensor()])

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
		return self.data_size

	def __getitem__(self, index):
		"""Retrieves images from folder and creates iterator"""

		# Load images
		inputimage = self.inputs[index]
		targetimage = self.targets[index]
		inpimg = tl.imread(inputimage)
		tarimg = tl.imread(targetimage)
		inpimg, tarimg = self._random_crop(inpimg, tarimg)

		inpimg = self.trans(self._norma(inpimg))
		tarimg = self.trans(self._norma(tarimg))

		return inpimg, tarimg
