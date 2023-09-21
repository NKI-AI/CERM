from pathlib import Path
from typing import Union

import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms


def create_cifar_datasets(
    path: Union[str, Path] = "./dataset/",
    random_train_val_split: bool = False,
    num_classes: int = 10,
):
    """
    Creates train,val and test data splits from mnist

    Parameters
    ----------
    random_train_val_split: bool
        if true, randomizes train val split, otherwise uses fixed seed
    num_classes: int
        number classes cifar dataset

    Returns
    -------
    train, val, test MNIST dataset

    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    if num_classes == 10:
        cifar_data = torchvision.datasets.CIFAR10
    elif num_classes == 100:
        cifar_data = torchvision.datasets.CIFAR100

    cifar_train_val = cifar_data(path, train=True, download=True, transform=transform)
    if random_train_val_split:
        train_dataset, val_dataset = random_split(cifar_train_val, [45000, 5000])
    else:
        train_dataset, val_dataset = random_split(
            cifar_train_val, [45000, 5000], torch.Generator().manual_seed(42)
        )

    test_dataset = cifar_data(path, train=False, download=True, transform=transform)

    return train_dataset, val_dataset, test_dataset
