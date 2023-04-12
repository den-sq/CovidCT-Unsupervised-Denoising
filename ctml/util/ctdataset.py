from enum import Enum
from pathlib import Path
import random

from empatches import EMPatches
from natsort import natsorted
import numpy as np
import tifffile as tf
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from util import log

emp = EMPatches()


class CTDataset(Dataset):
	def __init__(self, root_dir, normalize_over, batch_size, patch_size, weights):
		"""Initializes abstract dataset."""
		super().__init__()
		self.imgs = []
		self.root_dir = Path(root_dir)
		self.normalize_over = normalize_over
		self.__patch_size = patch_size
		self.__weights = weights
		self.__batch_size = batch_size
		self.trans = transforms.Compose([transforms.ToTensor()])

		if self.root_dir.exists():
			self.inputs = natsorted(self.root_dir.iterdir())
			log.log("Creating Dataset", len(self.inputs))

	def _norma(self, img, bottom_threshold, top_threshold):
		""" Basic Image Normalization. """
		floor = np.percentile(img, bottom_threshold)
		ceiling = np.percentile(img, top_threshold)

		if floor == ceiling:
			# log.log("Normalization", f"{floor}:{ceiling}:{np.max(img)}:{np.min(img)}", log_level=log.DEBUG.WARN)
			ceiling = np.max(img)
			floor = np.min(img)

		normalized = img - floor
		normalized[normalized < 0] = 0

		return normalized / (ceiling - floor)

	@property
	def patch_size(self):
		return self.__patch_size

	@property
	def weights(self):
		return self.__weights

	@property
	def batch_size(self):
		return self.__batch_size

	def __len__(self):
		"""Returns length of dataset."""
		return len(self.inputs)

	def __getitem__(self, index):
		"""Retrieves images from folder and creates iterator"""
		return self.trans(self.inputs[index])

	@staticmethod
	def dup(ds):
		return CTDataset(ds.root_dir, ds.normalize_over, ds.batch_size, ds.patch_size, ds.weights)


class CTDenoisingSet(CTDataset):
	def __init__(self, image, normalize_over, batch_size, patch_size, patch_overlap, weights):
		super().__init__(".", normalize_over, batch_size, patch_size, weights)
		self.__img = tf.imread(image)

		#  Trim (2D) for Patching matching Patch Size
		trims = [(dim % patch_size) // 2 for dim in self.__img.shape]
		self.__trim_index = np.s_[trims[0]:self.__img.shape[0] - trims[0], trims[1]:self.__img.shape[1] - trims[1]]

		# Generate Patches of Normalized Data
		if normalize_over is not None:
			norm_img = self._norma(self.__img[self.__trim_index], normalize_over.start, normalize_over.stop)
		else:
			norm_img = self.__img[self.trim_index]
		log.log("Image Normalization", f"{normalize_over}")

		self.inputs, self.__indices = emp.extract_patches(norm_img, patchsize=patch_size, overlap=patch_overlap)
		log.log("Patch Creation", f"Patches Created ({len(self.indices)})")

	@property
	def img(self):
		return self.__img

	@property
	def trim_index(self):
		return self.__trim_index

	@property
	def indices(self):
		return self.__indices

	@staticmethod
	def extract(ds, image, patch_overlap):
		return CTDenoisingSet(image, ds.normalize_over, ds.batch_size, ds.patch_size, patch_overlap, ds.weights)


class CTTrainingSet(CTDataset):
	def __init__(self, root_dir, normalize_over, batch_size, patch_size, weights):
		super().__init__(root_dir, normalize_over, batch_size, patch_size, weights)
		self.targets = self.inputs[1:] + [self.inputs[-2]]

	def _random_crop(self, image1, image2):
		m, n = image1.shape
		randx = random.randint(0, m - (self.patch_size + 5))
		randy = random.randint(0, n - (self.patch_size + 5))
		image1 = image1[randx:randx + self.patch_size, randy:randy + self.patch_size]
		image2 = image2[randx:randx + self.patch_size, randy:randy + self.patch_size]
		return image1, image2

	def __getitem__(self, index):
		"""Retrieves images from folder and creates iterator"""

		# Load images
		inputimage = self.inputs[index]
		targetimage = self.targets[index]
		inpimg = tf.imread(inputimage)
		tarimg = tf.imread(targetimage)
		inpimg, tarimg = self._random_crop(inpimg, tarimg)

		if self.normalize_over is not None:
			return (self.trans(self._norma(inpimg, self.normalize_over.start, self.normalize_over.stop)),
					self.trans(self._norma(tarimg, self.normalize_over.start, self.normalize_over.stop)))
		else:
			return self.trans(inpimg), self.trans(tarimg)

	@staticmethod
	def dup(ds):
		return CTTrainingSet(ds.root_dir, ds.normalize_over, ds.batch_size, ds.patch_size, ds.weights)


class FileSet(Enum):
	TRAIN = 0,
	VALIDATE = 1,
	FULL = 2,
	PATCHES = 3

	def load(self, ds: Dataset, batch_size=None, image=None, overlap=False, single=False, shuffle=False):
		if batch_size is None:
			batch_size = ds.batch_size
		if self == FileSet.PATCHES:
			ds_two = CTDenoisingSet.extract(ds, image, overlap)
			return DataLoader(ds_two, batch_size=1 if single else batch_size, shuffle=shuffle), ds_two
		else:
			dup = CTTrainingSet.dup(ds)
			if self == FileSet.TRAIN:
				return DataLoader(Subset(dup, range(0, int(len(dup) * 0.75))),
									batch_size=1 if single else batch_size, shuffle=shuffle)
			elif self == FileSet.VALIDATE:
				return DataLoader(Subset(dup, range(int(len(dup) * 0.75), len(ds))),
									batch_size=1 if single else batch_size, shuffle=shuffle)
			elif self == FileSet.FULL:
				return DataLoader(dup, batch_size=1 if single else batch_size, shuffle=shuffle)
