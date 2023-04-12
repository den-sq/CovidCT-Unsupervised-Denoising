import numpy as np

from util.ctdataset import emp
from util import log


class CTDenoiser(object):
	def __init__(self, model, use_cuda):
		self.__model = model
		self.__use_cuda = use_cuda

	def __apply_model(self, source):
		if self.__use_cuda:
			source = source.cuda()

		return tensortoimage(self.__model(source).detach())

	def denoise(self, patches, ds):
		# Denoise and Remerge Patches
		denoised_imgs = [self.__apply_model(source) for source in patches]
		log.log("Denoising", "Patches Denoised")

		# Write out patched image with original data in un-patched areas, as 32 bit.
		merged = emp.merge_patches(denoised_imgs, ds.indices, mode='avg')
		out_img = ds.img.astype(np.float32)
		out_img[ds.trim_index] = merged

		log.log("Merging", "Image Merged")
		return merged


def tensortoimage(tor):
	img = tor.cpu()
	img = img.squeeze(0)
	img = img.squeeze(0)
	return img.numpy()