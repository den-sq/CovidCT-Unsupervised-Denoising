from argparse import ArgumentParser

from customdataset import load_dataset
from ctdenoiser import CTdenoiser


def parse_args():
    """Command-line argument parser for training."""
    # New parser
    parser = ArgumentParser(description='Covid CT denoiser training script')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='/data/train_recons/')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/data/validation_recons/')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./model/ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=128, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=2, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l2', type=str)
    parser.add_argument('--cuda', help='will use cuda by default', action='store_false')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--add noise-type', help='noise type',
        choices=['natural', 'poisson', 'text', 'mc'], default='natural', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=512, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    """Trains CT denoiser."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(root_dir=params.train_dir, params=params, shuffled=True)
    valid_loader = load_dataset(root_dir=params.valid_dir, params=params, shuffled=False)

    # Initialize model and train
    ctd = CTdenoiser(params, trainable=True)
    ctd.train(train_loader, valid_loader)
