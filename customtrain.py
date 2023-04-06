import click

from customdataset import CTDataset
from ctdenoiser import CTDenoiser


@click.command()
# Data parameters
@click.option('-t', '--train-dir', type=click.Path(), default='/data/train_recons/',
	help='Path to training dataset')
@click.option('-v', '--valid-dir', type=click.Path(), default='/data/validation_recons/',
	help='Path to validation dataset')
@click.option('--ckpt-save-path', type=click.Path(), default='./model/ckpts', help='Checkpoint save path')
@click.option('--ckpt-overwrite', type=click.BOOL, default=False, help='Overwrite intermediate model checkpoints')
@click.option('--report-interval', type=click.INT, default=128, help='Batch report interval')
@click.option('--plot-stats/--skip-plotting', type=click.BOOL, default=True, help='plot stats after every epoch')
# Training hyperparameters
@click.option('-lr', '--learning-rate', type=click.FLOAT, default=0.001, help='learning rate')
@click.option('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
@click.option('-b', '--batch-size', type=click.INT, default=4, help='minibatch size')
@click.option('-e', '--nb-epochs', type=click.INT, default=2, help='number of epochs')
@click.option('-l', '--loss', type=click.Choice(['l1', 'l2']), default='l2', help='loss function')
@click.option('--cuda/--no-cuda', type=click.BOOL, help='will use cuda by default', default=False)
# Corruption parameters
@click.option('-n', '--noise-type', type=click.Choice(['natural', 'poisson', 'text', 'mc']), default='natural',
	help='Type of noise to target.')
@click.option('-c', '--crop-size', help='random crop size', default=512, type=int)
def unsupervised_training(train_dir, valid_dir, ckpt_save_path, ckpt_overwrite, report_interval, plot_stats,
							learning_rate, adam, batch_size, nb_epochs, loss, cuda, noise_type, crop_size):
	"""Trains CT denoiser."""
	# Train/valid datasets
	train_loader = CTDataset(train_dir, crop_size).loader(batch_size, shuffle=True)
	valid_loader = CTDataset(valid_dir, crop_size).loader(batch_size, shuffle=False)

	# Initialize model and train
	ctd = CTDenoiser(loss, noise_type, learning_rate, adam, nb_epochs, cuda, trainable=True)
	ctd.train(train_loader, valid_loader, report_interval, plot_stats, ckpt_save_path, ckpt_overwrite)


if __name__ == '__main__':
	unsupervised_training()
