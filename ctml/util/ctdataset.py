from collections import namedtuple
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

coord = namedtuple("coord", ["x", "y"])


class CTDataset(Dataset):
	def __init__(self, root_dir, normalize_over, batch_size, patch_size, weights):
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
		""" Basic image normalization steps. """
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
		""" Size of patches within an image to use for ML analysis. """
		return self.__patch_size

	@property
	def weights(self):
		""" Location of ML saved weights. """
		return self.__weights

	@property
	def batch_size(self):
		""" # of patches sent to GPU at once for multiprocessing. """
		return self.__batch_size

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, index):
		return self.trans(self.inputs[index])

	@staticmethod
	def dup(ds):
		""" Returns a CTDataset duplicate of passed in dataset. """
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
	def extract(ds: CTDataset, image, patch_overlap: float):
		""" Extracts a denoising dataset consisting of patches of an existing image.

			:param ds: Dataset image is from.
			:param image: Image to extract a patch set from.
			:param patch_overlap: Overlap value between patches.
		"""
		return CTDenoisingSet(image, ds.normalize_over, ds.batch_size, ds.patch_size, patch_overlap, ds.weights)


class CTTrainingSet(CTDataset):
	def __init__(self, root_dir, normalize_over, batch_size, patch_size, weights):
		super().__init__(root_dir, normalize_over, batch_size, patch_size, weights)
		self.targets = self.inputs[1:] + [self.inputs[-2]]

	def _random_crop(self, image1, image2):
		image_dim = coord(*image1.shape)
		pos = coord(random.randint(0, image_dim.x - (self.patch_size + 4)),
					random.randint(0, image_dim.y - (self.patch_size + 4)))
		crop = np.s_[pos.x:pos.x + self.patch_size, pos.y:pos.y + self.patch_size]
		return image1[crop], image2[crop]

	def __getitem__(self, index):
		"""Creates and returns GPU-located crops of the source and target images for the index.

			Returned images are cropped and normalized according to the dataset's batch size
			and normalization parameters.
		"""

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
		""" Creates a CTTrainingSet duplicate of a CTDataset. """
		return CTTrainingSet(ds.root_dir, ds.normalize_over, ds.batch_size, ds.patch_size, ds.weights)


class FileSet(Enum):
	""" Enum handling different kinds of file loads from a DataSet. """
	TRAIN = 0,
	VALIDATE = 1,
	FULL = 2,
	PATCHES = 3

	def load(self, ds: Dataset, image=None, overlap=False, single=False, shuffle=False):
		""" Loads a given dataset into a dataloader based on what kind of FileSet this is.

			:param ds: Dataset to load.
		"""
		if self == FileSet.PATCHES:
			ds_two = CTDenoisingSet.extract(ds, image, overlap)
			return DataLoader(ds_two, batch_size=1 if single else ds.batch_size, shuffle=shuffle), ds_two
		else:
			dup = CTTrainingSet.dup(ds)
			if self == FileSet.TRAIN:
				return DataLoader(Subset(dup, range(0, int(len(dup) * 0.75))),
									batch_size=1 if single else ds.batch_size, shuffle=shuffle)
			elif self == FileSet.VALIDATE:
				return DataLoader(Subset(dup, range(int(len(dup) * 0.75), len(ds))),
									batch_size=1 if single else ds.batch_size, shuffle=shuffle)
			elif self == FileSet.FULL:
				return DataLoader(dup, batch_size=1 if single else ds.batch_size, shuffle=shuffle)
