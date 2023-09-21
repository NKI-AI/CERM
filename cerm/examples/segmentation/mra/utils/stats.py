"""Export statistics multiple models."""

import torch
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from mra.network.model import ContourModel
from mra.dataset.dataset import ContourData
from mra.dataset.data_utils import construct_data_loader
from mra.metrics import metrics
from mra.utils.inference import move_sample_to_device, remove_corrupt_samples, predict


def compute_mean_midpoint(
    data_loader: DataLoader,
    device: str,
    num_spatial_comp: int = 2,
) -> Tensor:
    """
    Compute mean midpoint

    Parameters
    ----------
    dataset: ContourData
        dataset
    device: str
        device on which computations are performed

    Returns
    -------
    float-valued PyTorch tensor of size [2]
        mean midpoint
    """
    # Preallocation
    sum_midpoint = torch.zeros(num_spatial_comp).to(device)
    num_non_empty_samples = 0

    for sample in data_loader:
        # Clean batch
        remove_corrupt_samples(sample)
        if len(sample["img"]) == 0:
            continue
        move_sample_to_device(sample, device)

        # Only consider non-empty samples
        non_void_idx = torch.where(sample["is_non_empty"] == 1)[0]
        if len(non_void_idx) > 0:
            sum_midpoint += torch.sum(sample["midpoint_contour"][non_void_idx], dim=0)
            num_non_empty_samples += len(non_void_idx)

    return sum_midpoint.cpu() / num_non_empty_samples


def init_stats_dataset(
    cfg: DictConfig,
    out_stats: Path,
    train_data: ContourData,
    val_data: ContourData,
    test_data: ContourData,
) -> None:
    """
    Initialize mean statistics datasets.

    Parameters
    ----------
    cfg: DictConfig
        configuration details model and dataset
    out_stats: Path
        output dir statistics
    train_data: ContourData
        training data
    val_data: ContourData
        validation data
    test_data: ContourData
        test data
    """
    mean_midpoint_dir = out_stats / "mean_midpoint_train.pth"

    # Compute mean midpoint training set
    if mean_midpoint_dir.exists():
        mean_midpoint = torch.load(mean_midpoint_dir)
    else:
        _, train_loader = construct_data_loader(
            train_data,
            cfg.trainer.dataloader.batch_size,
            cfg.setup.num_workers,
            subsample_idx=train_data.non_empty_slices,
            prefetch_factor=cfg.trainer.dataloader.prefetch_factor,
            shuffle=False,
            persistent_workers=False,
        )
        mean_midpoint = compute_mean_midpoint(
            train_loader,
            cfg.setup.device,
        )
        torch.save(mean_midpoint, mean_midpoint_dir)

    # Set mean midpoint
    train_data.set_mean_midpoint(mean_midpoint)
    val_data.set_mean_midpoint(mean_midpoint)
    test_data.set_mean_midpoint(mean_midpoint)


def model_performance(
    model: ContourModel,
    dataset: ContourData,
    batch_size: int,
    out_dir: Path,
    device: str,
    num_spatial_comp: int = 2,
    prefetch_factor: int = 2,
    num_workers: int = 4,
) -> None:
    """
    Evaluate metrics trained model

    Parameters
    ----------
    model: ContourModel (subclass of nn.Module)
        neural network modeling wavelet decomposition of contour
    dataset: ContourData
        initialized dataset
    batch_size: int
        batch size dataloader
    out_dir: Path
        path to which results (scores) are written
    device: str
        device on which computations are performed
    prefetch_factor: int
        number of samples preloaded by each worker
    num_workers: int
        number of processed used by dataloader
    """
    if not out_dir.exists():
        out_dir.mkdir()

    # Dimensions
    dim_img = dataset.dim_img
    init_res_level = dataset.init_res_level
    mean_midpoint = dataset.mean_midpoint

    # Data loader
    _, data_loader = construct_data_loader(
        dataset,
        batch_size,
        num_workers,
        subsample_idx=dataset.non_empty_slices,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        persistent_workers=False,
    )

    # Preallocation
    contour_metrics = {"dice": [], "hausdorff": []}

    with torch.no_grad():
        # Perform inference over entire dataset first to keep GPU busy
        obs, pred = predict(model, data_loader, device)

        # Evaluate mean metrics
        num_batches = len(pred)
        num_samples = 0
        num_non_empty_samples = 0
        for obs_batch, pred_batch in zip(obs, pred):
            # Determine non-empty samples in batch
            is_non_empty = obs_batch["is_non_empty"]
            num_samples_in_batch = len(is_non_empty)
            non_empty_batch_idx = torch.where(is_non_empty == 1)[0]
            num_non_empty_in_batch = len(non_empty_batch_idx)
            if num_non_empty_in_batch == 0:
                continue

            # Update number of samples
            num_samples += num_samples_in_batch
            num_non_empty_samples += num_non_empty_in_batch

            # Groundtruth and predicted curves
            contour_gt = (
                2 ** (init_res_level / 2)
                * obs_batch["contour_dyadic_scaled"][non_empty_batch_idx]
                + mean_midpoint
            )
            contour_pred = (
                2 ** (init_res_level / 2)
                * pred_batch["approx"][-1][non_empty_batch_idx]
                + mean_midpoint
            )

            # Performance measures
            dice, hausdorff = metrics.discrepancy_contours(contour_gt, contour_pred)
            contour_metrics["dice"].append(dice)
            contour_metrics["hausdorff"].append(hausdorff)

    # Export
    for key, vals in contour_metrics.items():
        filename = out_dir / f"{key}"
        torch.save(torch.cat(vals), f"{str(filename)}.pth")
