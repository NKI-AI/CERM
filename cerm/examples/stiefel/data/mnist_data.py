from pathlib import Path
from typing import Union

import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms


def create_mnist_datasets(
    path: Union[str, Path] = "./dataset/", random_train_val_split: bool = False
):
    """
    Creates train,val and test data splits from mnist

    Parameters
    ----------
    random_train_val_split: bool
        if true, randomizes train val split, otherwise uses fixed seed

    Returns
    -------
    train, val, test MNIST dataset

    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    mnist_train_val = torchvision.datasets.MNIST(
        path, train=True, download=True, transform=transform
    )
    if random_train_val_split:
        train_dataset, val_dataset = random_split(mnist_train_val, [55000, 5000])
    else:
        train_dataset, val_dataset = random_split(
            mnist_train_val, [55000, 5000], torch.Generator().manual_seed(42)
        )

    test_dataset = torchvision.datasets.MNIST(
        path, train=False, download=True, transform=transform
    )

    return train_dataset, val_dataset, test_dataset
