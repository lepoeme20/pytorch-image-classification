"""Train module for classifier
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from classifier import classifier
from get_loader import get_dataloader

def train(args):
    """[summary]

    Arguments:
        args {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    save_root = os.path.join(args.pretrained_dir, args.dataset)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, args.classifier + '.pth')

    trainloader, devloader, _ = get_dataloader(args)

    net = classifier(
        args=args, clf=args.classifier, train=args.train,
        pretrained_dir=args.pretrained_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr_clf, momentum=0.9)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    if torch.cuda.is_available():
        print('\n===> Training on GPU!')
        net = nn.DataParallel(net)

    best_acc = 0
    for epoch in range(args.epochs):
        print('\n===> epoch %d' % epoch)
        running_loss = 0.0
        test_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            net.train()

            # get inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if args.dataset.lower() == 'celeba':
                labels = labels[:, 20]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 20 == 19:  # print every 100 mini-batches
                print('{}/{} loss:{:.3f}'.format(i + 1, len(trainloader) + 1, running_loss / 20))
                running_loss = 0.0

        for idx, dev in enumerate(devloader):
            imgs, labels = dev
            imgs, labels = imgs.to(device), labels.to(device)

            if args.dataset.lower() == 'celeba':
                labels = labels[:, 20]

            net.eval()
            with torch.no_grad():
                outputs = net(imgs)

                # Loss
                loss = criterion(outputs, labels)
                test_loss += loss

                # Accuracy
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                print('[Dev] {}/{} Loss: {:.3f}, Acc: {:.3f}'.format(
                    idx+1, len(devloader), test_loss/(idx+1), 100.*(correct/total)))


        # Save checkpoint
        acc = 100. * (correct/total)
        scheduler.step(test_loss)

        if acc > best_acc:
            print("Acc: {:.4f}".format(acc))
            best_acc = acc
            if torch.cuda.device_count() > 0:
                torch.save({
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

    return net
