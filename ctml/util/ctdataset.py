from collections import namedtuple
from enum import Enum
from multiprocessing.pool import ThreadPool
from pathlib import Path
import random

from empatches import EMPatches
from natsort import natsorted
import numpy as np
from psutil import cpu_count
import tifffile as tf
from tomopy import circ_mask
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from util import log
from util.util import FloatRange

emp = EMPatches()

coord = namedtuple("coord", ["x", "y"])


class CTDataset(Dataset):
    def __init__(self, root_dir, circ_mask_ratio, batch_size, patch_size, weights):
        super().__init__()
        self.imgs = []
        self.root_dir = Path(root_dir)

        self.__circ_mask_ratio = circ_mask_ratio
        self.__patch_size = patch_size
        self.__weights = weights
        self.__batch_size = batch_size
        self.__floor = None
        self.__ceiling = None
        self.trans = transforms.Compose([transforms.ToTensor()])

        if self.root_dir.exists():
            self.inputs = natsorted(self.root_dir.iterdir())
            log.log("Creating Dataset", len(self.inputs))

    def _get_tiff_detail(self, img):
        with tf.TiffFile(img) as tif:
            page = tif.pages[0]
            self._idim = coord(*page.shape)
            self._dtype = page.dtype
            self._bytesize = page.dtype.itemsize
            self._offset = page.dataoffsets[0]

    # ONLY USE WITH THREADS NOT PROCESSES TAKES IMAGE FILE
    def _mem_read(self, image, i, path):
        image[i] = tf.imread(path)

    def norm_setup(self, normalize_over):
        if self.circ_mask_ratio:
            self.__normalize_over = FloatRange(normalize_over.start + 100 * (1.0 - np.pi * ((self.circ_mask_ratio / 2) ** 2)),
                                normalize_over.start, normalize_over.step)
        else:
            self.__normalize_over = normalize_over

        self._get_tiff_detail(self.inputs[0])

        processes = cpu_count
        normalize_interval = int(np.ceil(len(self.inputs) / cpu_count))

        # Normalization calculation
        image_count = len(self.inputs) // normalize_interval
        log.log("Image Load", f"{len(self.inputs)}:{normalize_interval}:{processes}:{len(self.inputs) // normalize_interval}")

        image = np.empty((cpu_count,) + self._idim)

        with ThreadPool(processes) as pool:
            pool.starmap(self._mem_read, [(image, i, self.inputs[j]) for i, j in enumerate(range(0, len(self.inputs), normalize_interval))])
        
        log.log("Image Load",
            f"{len(range(0, len(self.inputs), len(self.inputs) % normalize_interval))}|{normalize_interval}"
            " Images Loaded For Normalization")

        if self.__circ_mask_ratio:
            circ_mask(image, axis=0, ratio=self.__circ_mask_ratio, val=np.min(image[0:image_count]))
            log.log("Image Masking", f"Images Masked at {self.__circ_mask_ratio}")

        self.__floor = np.percentile(image, self.__normalize_over.start)
        self.__ceiling = np.percentile(image, self.__normalize_over.stop)
        if self.__floor == self.__ceiling:
            self.__ceiling += 0.001

    def _norma(self, img):
        """ Basic image normalization steps. """
        normalized = img - self.__floor
        normalized[normalized < 0] = 0

        return normalized / (self.__ceiling - self.__floor)

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
    
    @property
    def floor(self):
        return self.__floor
    
    @property
    def ceiling(self):
        return self.__ceiling
    
    def set_range(self, floor, ceiling):
        self.__floor = floor
        self.__ceiling = ceiling

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.trans(self.inputs[index])

    @staticmethod
    def dup(ds):
        """ Returns a CTDataset duplicate of passed in dataset. """
        temp = CTDataset(ds.root_dir, ds.normalize_over, ds.batch_size, ds.patch_size, ds.weights)
        temp.setrange(ds.floor, ds.ceiling)
        return temp


class CTDenoisingSet(CTDataset):
    def __init__(self, image, floor, ceiling, batch_size, patch_size, patch_overlap, weights, preload):
        super().__init__(".", batch_size, patch_size, weights)
        self.set_range(floor, ceiling)
        self._get_tiff_detail(image)
        if preload is None:
            self.__img = tf.imread(image).astype(np.float32)
            log.log("Image Load", f"{image.name}")
        else:
            self.__img = preload.astype(np.float32)
            log.log("Image Preloaded", f"{image.name}")

        # Generate Patches of Normalized Data
        if floor is not None and ceiling is not None:
            self.__img = self._norma(self.__img)
            log.log("Image Normalization", f"{floor} to {ceiling}")

        self.inputs, self.__indices = emp.extract_patches(self.__img, patchsize=patch_size, overlap=patch_overlap)
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
    def extract(ds: CTDataset, image, patch_overlap: float, preload=None):
        """ Extracts a denoising dataset consisting of patches of an existing image.

            :param ds: Dataset image is from.
            :param image: Image to extract a patch set from.
            :param patch_overlap: Overlap value between patches.
            :param preload: Preloaded image file.
        """
        return CTDenoisingSet(image, ds.floor, ds.ceiling, ds.batch_size, ds.patch_size, patch_overlap, ds.weights, preload)


class CTTrainingSet(CTDataset):
    def __init__(self, root_dir, floor, ceiling, batch_size, patch_size, weights):
        super().__init__(root_dir, batch_size, patch_size, weights)
        self.set_range(floor, ceiling)
        self._get_tiff_detail(self.inputs[0])
        self.targets = self.inputs[1:] + [self.inputs[-2]]

    def _random_crop(self):
        pos = coord(random.randint(0, self._idim.x - (self.patch_size + 4)),
                    random.randint(0, self._idim.y - (self.patch_size + 4)))
        return np.s_[pos.x:pos.x + self.patch_size, pos.y:pos.y + self.patch_size]

    def __getitem__(self, index):
        """Creates and returns GPU-located crops of the source and target images for the index.

            Returned images are cropped and normalized according to the dataset's batch size
            and normalization parameters.
        """

        # Load images
        crop = self._random_crop()
        inpimg = np.memmap(self.inputs[index], self._dtype, 'r', self._offset, self._idim)[crop]
        tarimg = np.memmap(self.targets[index], self._dtype, 'r', self._offset, self._idim)[crop]

        if self.floor is not None and self.ceiling is not None:
            return (self.trans(self._norma(inpimg)), self.trans(self._norma(tarimg)))
        else:
            return self.trans(inpimg), self.trans(tarimg)

    @staticmethod
    def dup(ds):
        """ Creates a CTTrainingSet duplicate of a CTDataset. """
        return CTTrainingSet(ds.root_dir, ds.floor, ds.ceiling, ds.batch_size, ds.patch_size, ds.weights)


class FileSet(Enum):
    """ Enum handling different kinds of file loads from a DataSet. """
    TRAIN = 0,
    VALIDATE = 1,
    FULL = 2,
    PATCHES = 3

    def load(self, ds: Dataset, image=None, overlap=False, single=False, shuffle=False, preload=None):
        """ Loads a given dataset into a dataloader based on what kind of FileSet this is.

            :param ds: Dataset to load.
        """
        batch_size = 1 if single else ds.batch_size
        if self == FileSet.PATCHES:
            ds_two = CTDenoisingSet.extract(ds, image, overlap, preload)
            return DataLoader(ds_two, batch_size=batch_size, shuffle=shuffle), ds_two
        else:
            dup = CTTrainingSet.dup(ds)
            train_stop = max(int(len(dup) * 0.75), len(dup) - 128)
            if self == FileSet.TRAIN:
                return DataLoader(Subset(dup, range(0, train_stop)), batch_size=batch_size, shuffle=shuffle)
            elif self == FileSet.VALIDATE:
                return DataLoader(Subset(dup, range(train_stop, len(ds))), batch_size=batch_size, shuffle=shuffle)
            elif self == FileSet.FULL:
                return DataLoader(dup, batch_size=batch_size, shuffle=shuffle)
