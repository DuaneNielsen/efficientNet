import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision import transforms
import torchvision
from torch.nn.functional import cross_entropy
from rich.progress import Progress
from collections import deque
from statistics import mean
from torchlars import LARS
from argparse import ArgumentParser
import wandb
import efficient_net
import os

if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--lars', action='store_true', default=False)
    args.add_argument('--dev', action='store_true', default=False)
    args.add_argument('--device', type=str)
    args.add_argument('--seed', type=int)
    args.add_argument('--lr', type=float, default=1e-4)
    config = args.parse_args()

    if 'DEVICE' in os.environ:
        config.device = os.environ['DEVICE']

    wandb.init(project='EfficientNet_CIFAR10', config=config)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    eff_net = efficient_net.EfficientNet(version="b0", num_classes=10).to(config.device)
    if config.lars:
        optim = LARS(Adam(eff_net.parameters(), lr=config.lr))
    else:
        optim = Adam(eff_net.parameters(), lr=config.lr)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='~/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform_test)

    if config.dev:
        trainset = Subset(trainset, list(range(1000)))
        testset = Subset(testset, list(range(1000)))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # reached 9762

    with Progress() as progress:

        t_epoch = progress.add_task('[red] epoch ...', total=100)
        t_ds = progress.add_task('[magenta] dataset ...', total=len(trainset) / config.batch_size)
        t_test = progress.add_task('[blue] dataset ...', total=len(testset) / 100)
        losses = deque(maxlen=100)

        for epoch in range(100):
            progress.update(t_epoch, advance=1)
            progress.reset(t_ds)
            progress.reset(t_test)
            for images, labels in trainloader:
                images = images.to(config.device)
                labels = labels.to(config.device)

                classes = eff_net(images)
                loss = cross_entropy(classes, labels)
                losses.append(loss.item())
                progress.update(t_ds, advance=1, description=f'[magenta] {mean(losses):.5f}')

                optim.zero_grad()
                loss.backward()
                optim.step()

            correct = 0
            total = 0
            for images, labels in testloader:
                images = images.to(config.device)
                labels = labels.to(config.device)
                classes = eff_net(images)
                classes = torch.argmax(classes, dim=1)
                for label, cls in zip(classes, labels):
                    if label == cls:
                        correct += 1
                    total += 1
                progress.update(t_test, advance=1, description=f'[blue]{correct}/{total}')

            wandb.log({'correct': correct})

