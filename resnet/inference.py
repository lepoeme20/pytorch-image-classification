"""Test your trained model
"""
import torch
import torch.nn as nn
from get_loader import get_dataloader

import config
from classifier import classifier as clf

def inference(args):
    """Main function to inference your trained model
    """

    _, _, testloader = get_dataloader(args)
    total_data = len(testloader.dataset)

    net = clf(
        args=args, clf=args.classifier, train=args.train,
        pretrained_dir=args.pretrained_dir)

    net.to(args.device)
    if torch.cuda.device_count() > 0:
        net = nn.DataParallel(net)

    net.eval()
    correct = 0
    with torch.no_grad():
        for data in testloader:
            imgs, labels = data
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            outputs = net(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    print(
        'Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total_data))

if __name__ == "__main__":
    opt = config.get_config()
    print(opt)

    inference(opt)
