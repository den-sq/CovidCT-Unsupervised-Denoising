import click
from empatches import EMPatches
from network import UNet
import numpy as np
from pathlib import Path
import tifffile as tf
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
		DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def norma(img):
	""" Basic Image Normalization. """
	return (img - np.min(img)) / (np.max(img) - np.min(img))


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
def unsupervised_denoise(data_dir, output_dir, mdpt, cuda, patch_size, patch_overlap):
	images = Path(data_dir).iterdir()
	print("Total Images to Denoise:", len(images))

	Path(output_dir).mkdir(exists_ok=True)

	model = UNet(in_channels=1)

	use_cuda = torch.cuda.is_available() and cuda

	if use_cuda:
		print("Using GPU")
		model = model.cuda()
		model.load_state_dict(torch.load(mdpt))
		model.eval()
	else:
		print("CUDA GPU is Unavailable or Skipped by Request.")

	for image in images:
		print(f"Cleaning Image {image.name}")
		img = tf.imread(image)

		#  Trim (2D) for Patching matching Patch Size
		trims = [(dim % patch_size) // 2 for dim in img.shape]
		trim_index = np.s_[trims[0]:img.shape[0] - trims[0], trims[1]:img.shape[1] - trims[1]]

		# Generate Patches of Normalized Data
		img_patches, indices = emp.extract_patches(norma(img[trim_index]),
													patchsize=patch_size, overlap=patch_overlap)

		# Denoise and Remerge Patches
		denoised_imgs = [denoise(source, model, use_cuda) for source in TensorTransformMapping(img_patches)]
		merged = emp.merge_patches(denoised_imgs, indices, mode='avg')

		# Write out patched image with original data in un-patched areas, as 32 bit.
		out_img = img.astype(np.float32)
		out_img[trim_index] = merged
		out_path = Path(output_dir, f"CL_{image.name}")
		print(f"Saving To {out_path}")
		tf.imsave(out_path, out_img)


if __name__ == "__main__":
	unsupervised_denoise()
