import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torch.optim import Adam
from torchvision import transforms
from torch.nn.functional import cross_entropy
from rich.progress import Progress
from collections import deque
from statistics import mean
from torchlars import LARS
from argparse import ArgumentParser
import wandb
import efficient_net

if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--lars', action='store_true', default=False)
    args.add_argument('--device', type=str)
    args.add_argument('--seed', type=int)
    args.add_argument('--lr', type=float, default=1e-4)
    config = args.parse_args()

    wandb.init(project='EfficientNet_MNIST', config=config)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    eff_net = efficient_net.EfficientNet(version="b0", num_classes=10).to(config.device)
    if config.lars:
        optim = LARS(Adam(eff_net.parameters(), lr=config.lr))
    else:
        optim = Adam(eff_net.parameters(), lr=config.lr)

    ds = MNIST('~/.mnist', train=True, download=True, transform=transforms.ToTensor())
    ds = Subset(ds, list(range(1000)))
    dl = DataLoader(ds, batch_size=config.batch_size)

    test_s = MNIST('~/.mnist', train=False, download=True, transform=transforms.ToTensor())
    test = DataLoader(test_s, batch_size=config.batch_size)

    # reached 9762

    with Progress() as progress:

        t_epoch = progress.add_task('[red] epoch ...', total=100)
        t_ds = progress.add_task('[magenta] dataset ...', total=len(ds) / config.batch_size)
        t_test = progress.add_task('[blue] dataset ...', total=len(test_s) / config.batch_size)
        losses = deque(maxlen=100)

        for epoch in range(100):
            progress.update(t_epoch, advance=1)
            progress.reset(t_ds)
            progress.reset(t_test)
            for images, labels in dl:
                images = images.to(config.device).expand(-1, 3, -1, -1)
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
            for images, labels in test:
                images = images.to(config.device).expand(-1, 3, -1, -1)
                labels = labels.to(config.device)
                classes = eff_net(images)
                classes = torch.argmax(classes, dim=1)
                for label, cls in zip(classes, labels):
                    if label == cls:
                        correct += 1
                    total += 1
                progress.update(t_test, advance=1, description=f'[blue]{correct}/{total}')

            wandb.log({'correct': correct})

