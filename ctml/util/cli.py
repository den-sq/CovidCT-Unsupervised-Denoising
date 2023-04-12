from pathlib import Path

import click
import numpy as np
from ruamel.yaml import YAML, yaml_object
import tifffile as tf
import torch

from unsup import ctnetwork, ctdenoise, cttrainer
from util import log
from util.ctdataset import CTDataset, FileSet

yaml = YAML()


@yaml_object(yaml)
class FloatRange:
	start: float
	stop: float
	step: float
	yaml_tag = '!FloatRange'

	def __init__(self, start, stop, step):
		self.start = start
		self.stop = stop
		self.step = step

	@classmethod
	def to_yaml(cls, representer, node):
		return representer.represent_scalar(cls.yaml_tag, str(node))

	@classmethod
	def from_yaml(cls, constructor, node):
		return cls(*node.value.split(","))

	def __str__(self):
		return f"{self.start},{self.stop},{self.step}"

	def as_array(self):
		steps = int((self.start - self.stop) // self.step) + 1
		return np.linspace(self.start, self.stop, steps)


# Click Parameter: Float Range (Imitated by linspace).
class Frange(click.ParamType):
	name = "Float Range"

	def convert(self, value, param, ctx):
		try:
			params = [float(x) for x in str(value).split(",")]
			start, stop, step = ([0.] if len(params) == 1 else []) + params + ([1.] if len(params) in [1, 2] else [])
			return FloatRange(start, stop, step)
		except ValueError:
			self.fail(f'{value} cannot be evaluated as a float range.')


FRANGE = Frange()


@click.group(chain=True)
@click.pass_context
@click.option('-d', '--data-dir', type=click.Path(), help='Input path for noisy dataset')
@click.option('-w', '--weights', type=click.Path(), help='Path to stored saved weights', default='data/weights.pt')
@click.option('-n', '--normalize-over', type=FRANGE, default=None,
				help="Range of retained values to normalize over, by percentiles.")
@click.option('-p', '--patch_size', type=click.INT, help="Size of image patches for analysis.", default=512)
@click.option('-b', '--batch-size', type=click.INT, default=4, help='# of Images for CUDA to batch process at once.')
@click.option('--cuda/--no_cuda', type=click.BOOL, help='Whether to use CUDA', default=False)
def ctml(ctx, data_dir, normalize_over, batch_size, patch_size, weights, cuda):
	if data_dir is None:
		pass
	elif Path(data_dir).exists():
		log.start()

		ctx.obj = CTDataset(data_dir, normalize_over, batch_size, patch_size, weights)
		ctnetwork.use_cuda = torch.cuda.is_available() and cuda

		if ctnetwork.use_cuda:
			log.log("Initialize", "Using GPU")
		elif cuda:
			log.log("Initialize", "CUDA GPU Skipped by Request.")
		else:
			log.log("Initialize", "CUDA GPU Unavailable.")
	else:
		log.log("Initialize", "Data Dir Does Not Exist", log.DEBUG.ERROR)


@ctml.command()
# Data parameters
@click.option('-v', '--valid-dir', type=click.Path(), default=None,
				help='Path to validation folder for non-automatic validation specification.')
@click.option('--ckpt-save-path', type=click.Path(), default='./data/ckpts', help='Checkpoint save path')
@click.option('--ckpt-overwrite', type=click.BOOL, default=False, help='Overwrite intermediate model checkpoints')
@click.option('--report-interval', type=click.INT, default=128, help='Batch report interval')
@click.option('--plot-stats/--skip-plotting', type=click.BOOL, default=True, help='Whether to plot stats after epochs')
# Training hyperparameters
@click.option('-lr', '--learning-rate', type=click.FLOAT, default=0.001, help='learning rate')
@click.option('-a', '--adam', type=click.FLOAT, multiple=True, default=[0.9, 0.99, 1e-8], help='adam parameters')
@click.option('-e', '--nb-epochs', type=click.INT, default=2, help='Epoch count')
@click.option('-l', '--loss', type=click.Choice(['l1', 'l2']), default='l2', help='Loss function')
# Corruption parameters
@click.option('-n', '--noise-type', type=click.Choice(['natural', 'poisson', 'text', 'mc']), default='natural',
	help='Type of noise to target.')
@click.pass_context
def utraining(ctx, valid_dir, ckpt_save_path, ckpt_overwrite, report_interval, plot_stats,
							learning_rate, adam, nb_epochs, loss, noise_type):
	"""Trains an Unsupervised ML Denoiser based on Noise2Noise ()"""

	# Load training and validation datasets
	if valid_dir is None:
		training_set = FileSet.TRAIN.load(ctx.obj, shuffle=True)
		validation_set = FileSet.VALIDATE.load(ctx.obj, shuffle=False)
	else:
		training_set = FileSet.FULL.load(ctx.obj, shuffle=True)
		validation_set = FileSet.FULL.load(
			CTDataset(valid_dir, ctx.obj.normalize_over, ctx.obj.batch_size. ctx.obj.patch_size, ctx.obj.weights),
			shuffle=False)

	# Initialize model and train
	ctd = cttrainer.CTTrainer(loss, noise_type, learning_rate, adam, nb_epochs, ctnetwork.use_cuda, trainable=True)
	ctd.train(training_set, validation_set, report_interval, plot_stats, ckpt_save_path, ckpt_overwrite)
	torch.save(ctd.model.state_dict(), ctx.obj.weights)


@ctml.command()
@click.option('-o', '--output-dir', type=click.Path(), help='Output path for cleaned images', default='data/clean/')
@click.option("--patch_overlap", type=click.FLOAT, help="Overlap between denoising patches", default=0.4)
@click.pass_context
def udenoise(ctx, output_dir, patch_overlap):
	"""Applies an Unsupervised ML Denoiser based on Noise2Noise ()"""
	log.log("Initialize Denoise", f"Total Images to Denoise: {len(ctx.obj.inputs)}")

	Path(output_dir).mkdir(exist_ok=True)

	model = ctnetwork.UNet(in_channels=1)

	if ctnetwork.use_cuda:
		model = model.cuda()
		model.load_state_dict(torch.load(ctx.obj.weights))
		model.eval()

	denoiser = ctdenoise.CTDenoiser(model, ctnetwork.use_cuda)

	for image in ctx.obj.inputs:
		log.log("Pass Start", f"Image {image.name}")

		# Load image and create patches.  Normalizes if needed.
		patches, ds = FileSet.PATCHES.load(ctx.obj, image=image, overlap=patch_overlap)

		# Denoises patches and merges back into original image, returning.
		out_img = denoiser.denoise(patches, ds)

		out_path = Path(output_dir, f"CL_{image.name}")
		tf.imwrite(out_path, out_img)
		log.log("Pass Complete", f"Image Saved To {out_path}")
