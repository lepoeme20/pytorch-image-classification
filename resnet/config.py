"""Set default hyper-parameter
"""
import argparse
import multiprocessing
import torch


def str2bool(v):
    """ convert str to boolean
    Arguments:
        v {str} -- "true" or "1"

    Returns:
        boolean -- if input is "ture" or "1", return True
    """
    return v.lower() in ("true", "1")


def default_parser_setting(parser):
    """Set default arguments
    """
    default_arg = parser.add_argument_group('Default')
    default_arg.add_argument(
        '--batch-size', type=int, default=512, help="mini-batch size for classification"
    )
    default_arg.add_argument(
        '--image-size', type=int, default=32, help="Image size"
    )
    default_arg.add_argument(
        '--image-channels', type=int, default=3, help="Image channels (RGB or B&W)"
    )
    default_arg.add_argument(
        '--n-workers', type=int, default=multiprocessing.cpu_count()-1,
        help="# of workers for dataloader"
    )
    default_arg.add_argument(
        '--save-dir', default="./pretrained_models/", type=str,
        help='saving path to pretrained model'
    )
    default_arg.add_argument(
        '--lr-clf', type=float, default=0.01, help='learning rate for classifier'
    )
    default_arg.add_argument(
        '--epochs', type=int, default=200, help="# of epochs for training classifier"
    )
    default_arg.add_argument(
        '--classifier', type=str, default='resnet18', help='[resnet18, resnet50, resnet101]'
    )
    default_arg.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['mnist', 'cifar10', 'fmnist', 'celeba']
    )
    default_arg.add_argument(
        '--data-root-path', type=str, default='datasets',
        help="The path where you want to save the data"
    )
    default_arg.add_argument(
        "--num-classes", type=int, default=10, help="number of total classes"
    )
    default_arg.add_argument(
        "--device-ids", type=int, nargs='*', help="Set GPU id(s)"
    )
    default_arg.add_argument(
        "--train", type=str2bool, help="train phase or test phase"
    )
    return parser


def get_config():
    """
    Returns:
        parser
    """
    parser = argparse.ArgumentParser()
    default_parser = default_parser_setting(parser)
    args, _ = default_parser.parse_known_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return args
