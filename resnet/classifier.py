"""
Set Classifier for training your dataset
"""
import os
import torch
from model import resnet18, resnet50, resnet101

def classifier(args, clf, train, pretrained_dir="/"):
    """Set classifier

    Arguments:
        dataset -- "Cifar10" or "Cifar100"
        clf -- "resnet18", "resnet50", or "resnet101"
        train {bool} -- Train or not

    Keyword Arguments:
        pretrained_dir {str} -- pretrained weights path (default: {"/"})

    Returns:
        Model
    """
    num_classes = args.num_classes
    input_channels = args.image_channels

    map_location = (lambda s, _: s)
    checkpoint_dir = os.path.join(pretrained_dir, args.dataset, clf + '.pth')

    if clf.lower() == 'resnet18':
        if train:
            net = resnet18(num_classes, input_channels)
        else:
            checkpoint = torch.load(checkpoint_dir, map_location=map_location)
            net = resnet18(num_classes, input_channels)
            net.load_state_dict(checkpoint['model_state_dict'])

    elif clf.lower() == 'resnet50':
        if train:
            net = resnet50(num_classes, input_channels)
        else:
            checkpoint = torch.load(checkpoint_dir, map_location=map_location)
            net = resnet50(num_classes, input_channels)
            net.load_state_dict(checkpoint['model_state_dict'])

    elif clf.lower() == 'resnet101':
        if train:
            net = resnet101(num_classes, input_channels)
        else:
            checkpoint = torch.load(checkpoint_dir, map_location=map_location)
            net = resnet101(num_classes, input_channels)
            net.load_state_dict(checkpoint['model_state_dict'])

    else:
        raise Exception("You can choose the model among [resnet18, resnet 50, resnet101]")

    return net
