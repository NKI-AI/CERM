"""Initialize dataloaders."""

import torch
import torchvision
import logging
import shutil
import os

from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from mra.dataset.dataset import ContourData, custom_collate


def slurm_copy_data(cfg: DictConfig, logger: logging.Logger) -> None:
    """
    Copy local data to scratch dir slurm

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict
    logger: logging.Logger
        module logger
    """
    current_dir = Path(os.getcwd())
    scratch_data_dir = Path(cfg.setup.scratch_dir) / "mra" / current_dir.stem / "data"
    if not scratch_data_dir.exists():
        scratch_data_dir.mkdir(parents=True)

    local_data_dirs = {
        "train": Path(cfg.dataset.train_dir),
        "val": Path(cfg.dataset.val_dir),
        "test": Path(cfg.dataset.test_dir),
    }

    for _, local_dir in local_data_dirs.items():
        logger.info(f"Copy data to scratch: {local_dir}")
        shutil.copytree(
            local_dir, scratch_data_dir / local_dir.stem, dirs_exist_ok=True
        )

    logger.info("Update directories in config")  # Dangerous?
    cfg.dataset.train_dir = scratch_data_dir / local_data_dirs["train"].stem
    cfg.dataset.val_dir = scratch_data_dir / local_data_dirs["val"].stem
    cfg.dataset.test_dir = scratch_data_dir / local_data_dirs["test"].stem


def init_dataset(
    cfg: DictConfig,
    data_dir: Path,
    preprocess: torchvision.transforms.Compose,
    data_aug=[],
) -> ContourData:
    """
    Initialize training and validation dataset.

    Parameters
    ----------
    cfg: DictConfig
        configuration details dataset
    data_dir: Path
        path to data (h5 files)
    preprocess: Torchvision compose object
        composition preprocessing functions
    data_aug: DataAug
        random augmentations

    Returns
    -------
    data: ContourData
        initialized dataset
    """
    return ContourData(
        data_dir,
        cfg.dataset.img_key,
        cfg.dataset.mask_key,
        cfg.dataset.dim_img,
        cfg.network.decoder.init_res_level,
        cfg.prep.order_fourier,
        resize=cfg.prep.resize,
        preprocess=preprocess,
        transform=data_aug,
    )


def construct_data_loader(
    dataset: ContourData,
    batch_size: int,
    num_workers: int,
    subsample_idx: List[int] = None,
    shuffle: bool = True,
    sampler=None,
    prefetch_factor: int = None,
    persistent_workers: bool = True,
) -> Tuple[int, DataLoader]:
    """
    Construct dataloader.

    Parameters
    ----------
    dataset: ContourData
        dataset for which a dataloader will be constructed
    batch_size: int
        number of samples used in each iteration
    num_workers: int
        number of processes used for multiprocessing
    subsample_idx: List[int]
        indices of subsamples to be used in sampler
    shuffle: bool
        shuffle dataset after every epoch
    prefetch_factor: int
        number of samples preloaded by each worker

    Returns
    -------
    num_samples: int
        number of samples in dataloader
    dataloader: DataLoader
        dataloader associated to dataset
    """
    if subsample_idx is not None:
        dataset = torch.utils.data.Subset(dataset, subsample_idx)
        num_samples = len(subsample_idx)
    else:
        num_samples = len(dataset)

    if prefetch_factor:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate,
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            collate_fn=custom_collate,
        )

    return num_samples, data_loader
