from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import tifffile as tf
import torch

from unsup import ctnetwork, ctdenoise, cttrainer
from util import log
from util.ctdataset import CTDataset, FileSet
from util.util import FloatRange, logged_write, thread_stub


# Click Parameter: Float Range (Imitated by linspace).
class Frange(click.ParamType):
    """ click parameter type for float ranges. """
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
@click.option('-p', '--patch-size', type=click.INT, help="Size of image patches for analysis.", default=512)
@click.option('-b', '--batch-size', type=click.INT, default=4, help='# of Images for CUDA to batch process at once.')
@click.option('-o', '--nodes', type=click.INT, default=1, help='# of GPU Nodes')
@click.option('--cuda/--no-cuda', type=click.BOOL, help='Whether to use CUDA', default=False)
def ctml(ctx, data_dir, normalize_over, batch_size, patch_size, weights, nodes, cuda):
    """ Applies ML methods to CT Data."""
    if data_dir is None:
        pass
    elif Path(data_dir).exists():
        log.start()

        ctx.obj = CTDataset(data_dir, normalize_over, batch_size, patch_size, weights)
        log.log("Initialize", f"{data_dir}")

        ctnetwork.use_cuda = torch.cuda.is_available() and cuda

        if ctnetwork.use_cuda:
            log.log("Initialize", "Using GPU")
            ctnetwork.nodes = min(nodes, torch.cuda.device_count())
            if ctnetwork.nodes < nodes:
                log.log("Initialize", f"Only {ctnetwork.nodes} GPUs available, cannot use {nodes} nodes.")
        elif cuda:
            log.log("Initialize", "CUDA GPU Unavailable", log_level=log.DEBUG.WARN)
        else:
            log.log("Initialize", "CUDA GPU Skipped by Request.", log_level=log.DEBUG.WARN)
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
    """Trains an Unsupervised ML Denoiser based on Noise2Noise () """

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
@click.option("--patch-overlap", type=click.FLOAT, help="Overlap between denoising patches", default=0.4)
@click.pass_context
def udenoise(ctx, output_dir, patch_overlap):
    """Applies an Unsupervised ML Denoiser based on Noise2Noise ()

        Ignores parent batch-size parameter due to empatches limitations.
        """

    log.log("Initialize Denoise", f"Total Images to Denoise: {len(ctx.obj.inputs)}")

    Path(output_dir).mkdir(exist_ok=True)
    log.log("Out Dir Created", output_dir)

    model = ctnetwork.UNet(in_channels=1)

    if ctnetwork.use_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(ctx.obj.weights))
        model.eval()
        if ctnetwork.nodes > 1:
            model = torch.nn.DataParallel(model)

    denoiser = ctdenoise.CTDenoiser(model, ctnetwork.use_cuda)

    with ThreadPoolExecutor(max_workers=3) as pool:
        # Thread Objects, pre-assigned as result() is called before first submit.
        # Don't preload first image because it may be skipped.
        writer = pool.submit(thread_stub)
        reader = pool.submit(thread_stub)

        for i, image in enumerate(ctx.obj.inputs):
            out_path = Path(output_dir, f"CL_{image.name}")
            if out_path.exists():
                log.log("Output Exists", out_path.name, log_level=log.DEBUG.WARN)
                continue

            log.log("Pass Start", f"Image {image.name}")

            # Load image and create patches.  Normalizes if needed.
            preload = reader.result()
            patches, ds = FileSet.PATCHES.load(ctx.obj,
                            single=True, image=image, overlap=patch_overlap, preload=preload)
            reader = pool.submit(tf.imread, ctx.obj.inputs[i + 1])

            # Make sure previous is written before overwriting.
            writer.result()

            # Denoises patches and merges back into original image, returning merged image.
            out_img = denoiser.denoise(patches, ds)

            writer = pool.submit(logged_write, out_path, out_img)
            log.log("Pass Complete", f"Image {i + 1} of {len(ctx.obj.inputs)}")
