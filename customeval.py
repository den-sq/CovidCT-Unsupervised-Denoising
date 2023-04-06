import click
from empatches import EMPatches
import numpy as np
from pathlib import Path
import tifffile as tf
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from network import UNet
import log

emp = EMPatches()


class TensorTransformMapping(Dataset):
	def __init__(self, listofimgs):
		self.list_of_images = listofimgs
		self.trans = transforms.Compose([transforms.ToTensor()])

	def __len__(self):
		""" Number of Images. """
		return len(self.list_of_images)

	def __getitem__(self, index):
		""" Returns Tensor Transformed Version of Image. """
		return self.trans(self.list_of_images[index])

	def loader(self, batch_size=1, shuffle=False):
		""" Gets Dataloader for Images """
		return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def norma(img, bottom_threshold, top_threshold):
	""" Basic Image Normalization. """
	floor = np.percentile(img, bottom_threshold)
	ceiling = np.percentile(img, 100.0 - top_threshold)

	normalized = img - floor
	normalized[normalized < 0] = 0

	return normalized / (ceiling - floor)


def tensortoimage(tor):
	img = tor.cpu()
	img = img.squeeze(0)
	img = img.squeeze(0)
	img = img.numpy()
	return img


def denoise(source, model, use_cuda):
	if use_cuda:
		source = source.cuda()
	# Denoise
	return tensortoimage(model(source).detach())


@click.command()
@click.option('--data-dir', type=click.Path(), help='Input path for noisy dataset', required=True)
@click.option('--output-dir', type=click.Path(), help='Output path for cleaned images', default='data/clean/')
@click.option('--mdpt', type=click.Path(), help='Path to stored saved weights', default='data/denoiser.pt')
@click.option('--cuda/--no_cuda', type=click.BOOL, help='Whether to use CUDA', default=False)
@click.option("--patch_size", type=click.INT, help="Size of denoising patches", default=256)
@click.option("--patch_overlap", type=click.FLOAT, help="Overlap between denoising patches", default=0.4)
@click.option("--bottom_threshold", type=click.FLOAT,
									help="Percentile threshold for bottom of normalization for noise removal", default=15.0)
@click.option("--top_threshold", type=click.FLOAT,
									help="Percentile threshold for top of normalization for container removal", default=1.0)
def unsupervised_denoise(data_dir, output_dir, mdpt, cuda, patch_size, patch_overlap, bottom_threshold, top_threshold):
	images = list(Path(data_dir).iterdir())
	log.log("Initialize", f"Total Images to Denoise: {len(images)}")

	Path(output_dir).mkdir(exist_ok=True)

	model = UNet(in_channels=1)

	use_cuda = torch.cuda.is_available() and cuda

	if use_cuda:
		log.log("Initialize", "Using GPU")
		model = model.cuda()
		model.load_state_dict(torch.load(mdpt))
		model.eval()
	elif cuda:
		log.log("Initialize", "CUDA GPU Skipped by Request.")
	else:
		log.log("Initialize", "CUDA GPU Unavailable.")

	for image in images:
		log.log("Pass Start", f"Image {image.name}")
		img = tf.imread(image)

		#  Trim (2D) for Patching matching Patch Size
		trims = [(dim % patch_size) // 2 for dim in img.shape]
		trim_index = np.s_[trims[0]:img.shape[0] - trims[0], trims[1]:img.shape[1] - trims[1]]

		# Generate Patches of Normalized Data
		norm_img = norma(img[trim_index], bottom_threshold, top_threshold)
		log.log("Image Normalization", f"B: {bottom_threshold} T: {top_threshold}")

		img_patches, indices = emp.extract_patches(norm_img, patchsize=patch_size, overlap=patch_overlap)
		patch_loader = TensorTransformMapping(img_patches).loader()
		log.log("Patch Creation", f"Patches Created ({len(indices)})")

		# Denoise and Remerge Patches
		denoised_imgs = [denoise(source, model, use_cuda) for source in patch_loader]
		log.log("Denoising", "Patches Denoised")

		# Write out patched image with original data in un-patched areas, as 32 bit.
		merged = emp.merge_patches(denoised_imgs, indices, mode='avg')
		out_img = img.astype(np.float32)
		out_img[trim_index] = merged
		out_path = Path(output_dir, f"CL_t{top_threshold}b{bottom_threshold}_{patch_size}_{image.name}")
		log.log("Pass Complete", f"Image Saved To  {out_path}")
		tf.imwrite(out_path, out_img)


if __name__ == "__main__":
	unsupervised_denoise()
