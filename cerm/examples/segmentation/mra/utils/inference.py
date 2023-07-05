"""Tools for exporting predictions."""

# TODO: refactor into smaller pieces

import logging

# import copy
import torch
import numpy as np
import matplotlib

from pathlib import Path
from typing import List, Tuple, Dict, Union
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from mra.dataset.dataset import ContourData, custom_collate
from mra.dataset.data_utils import construct_data_loader
from mra.network.model import ContourModel
from mra.network import model_utils
from mra.utils.plot_utils import plot_contour, plot_dwt_contour, plot_wavelets

# Set up module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def move_sample_to_device(
    sample: Dict[str, Union[Tensor, List[Tensor]]], device: str
) -> None:
    """
    Transfer relevant components of a sample to device.

    Parameters
    ----------
    sample: Dict[str, tensor]
        sample containing approximation and detail coefficients and image
    device: str
        device to which approx, detail and image are moved
    """
    for key in [
        "img",
        "mask",
        "is_non_empty",
        "midpoint_contour",
        "contour_dyadic_scaled",
    ]:
        sample[key] = sample[key].to(device)


def remove_corrupt_samples(sample: Dict[str, Union[Tensor, List[Tensor]]]) -> None:
    """
    Remove corrupt samples from batch

    Parameters
    ----------
    sample: Dict[str, Union[Tensor, List[Tensor]]]
        sample containing image, mask, approximation and detail coefficients
    """
    non_corrupt_idx = torch.where(sample["is_corrupt"] == 0)[0]
    for sample_key, sample_value in sample.items():
        if sample_key == "contour":
            sample[sample_key] = [sample_value[idx] for idx in non_corrupt_idx]
        else:
            sample[sample_key] = sample_value[non_corrupt_idx]


def predict(
    model: ContourModel,
    data_loader: DataLoader,
    device: str,
    ignore_corrupt_samples: bool = True,
) -> Tuple[
    List[Dict[str, Union[Tensor, List[Tensor]]]],
    List[Dict[str, Union[Tensor, List[Tensor]]]],
]:
    """
    Perform inference on dataset.

    Parameters
    ----------
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contours
    data_loader: DataLoader
        dataloader contours
    device: str
        device on which computations are performed
    ignore_corrupt_samples: bool
        indicates whether to perform inference on corrupt samples or not

    Returns
    -------
    obs: List[Dict[str, Union[Tensor, List[Tensor]]]]
        observations
    pred: List[Dict[str, Union[Tensor, List[Tensor]]]]
        predicted wavelet coefficients and probability maps
    """
    # Preallocation
    pred = []
    obs = []

    # Store predictions and ground-truth
    model.eval()
    with torch.no_grad():
        for sample in data_loader:
            # Only consider non-corrupted ground-truth
            if ignore_corrupt_samples:
                remove_corrupt_samples(sample)
                if len(sample["img"]) == 0:
                    continue

            # Predict and move to cpu
            pred_batch = model(sample["img"].to(device))

            # Classification and segmentation
            for key in ["classifier"]:
                pred_batch[key] = pred_batch[key].cpu()

            # Approximation coefficients
            for level in range(len(pred_batch["approx"])):
                pred_batch["approx"][level] = pred_batch["approx"][level].cpu()

            # Detail coefficients
            for level in range(len(pred_batch["detail"])):
                pred_batch["detail"][level] = pred_batch["detail"][level].cpu()

            pred.append(pred_batch)
            obs.append(sample)

    return obs, pred


def export_predictions(
    model: ContourModel,
    dataset: ContourData,
    out_dir: Path,
    num_workers: int = 4,
    batch_size: int = 16,
    device: str = "cpu",
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 100,
    plot_preprocessed: bool = True,
) -> None:
    """
    Plot predicted and observed contour and write to tensorboard.

    Parameters
    ----------
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contours
    dataset: ContourData
        dataset
    out_dir: Path
        output path to store figures in
    num_workers: int
        number of workers
    batch_size: int
        batch_size
    device: str
        device on which computations are performed
    figsize: Tuple[int, int]
        figure size
    dpi: int
        dots per inch
    plot_preprocessed: bool
        plot preprocessed data if true
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Prepare dataset and dataloader
    dataset.transform = None
    _, data_loader = construct_data_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        persistent_workers=False,
    )

    # Perform inference
    obs, pred = predict(model, data_loader, device, ignore_corrupt_samples=False)

    # Export volumes
    if not plot_preprocessed:
        if dataset.resize:
            dataset.preprocess.transforms = [
                dataset.preprocess.transforms[0]
            ]  # hacky but ok
        else:
            dataset.preprocess = None

    for img_idx, img_dir in enumerate(dataset.img_paths):
        logger.info(f"Exporting {img_dir}")

        curr_folder = Path(img_dir).parent
        out_folder = out_dir / curr_folder.relative_to(dataset.data_dir)
        out_dir_img = out_folder / "img"
        out_dir_coeffs = out_folder / "coeffs"
        if not out_dir_img.exists():
            out_dir_img.mkdir(parents=True)
        if not out_dir_coeffs.exists():
            out_dir_coeffs.mkdir(parents=True)

        # Determine slice indices associated to image
        if img_idx == 0:
            subidx = range(dataset.cum_num_slices[img_idx])
        else:
            subidx = range(
                dataset.cum_num_slices[img_idx - 1],
                dataset.cum_num_slices[img_idx],
            )

        for slice_idx in subidx:
            # Get slice info
            sample = dataset[slice_idx]
            batch_idx = slice_idx // batch_size
            slice_idx_rel_to_batch = slice_idx % batch_size

            # Plot if non-empty
            if sample["is_non_empty"] and not sample["is_corrupt"]:
                fig_contour, ax_contour = plot_contour(
                    sample["img"].squeeze(0),
                    sample["contour_dyadic_scaled"],
                    pred[batch_idx]["approx"][-1][slice_idx_rel_to_batch],
                    dataset.init_res_level,
                    figsize,
                    avg_midpoint_contour=dataset.mean_midpoint,
                )

                fig_contour.savefig(
                    f"{out_dir_img / str(slice_idx)}.pdf",
                    dpi=dpi,
                    bbox_inches="tight",
                )
                plt.close(fig_contour)

                fig_dwt, ax_dwt = plot_dwt_contour(
                    sample["contour_dyadic_scaled"],
                    [a[slice_idx_rel_to_batch] for a in pred[batch_idx]["approx"]],
                    [d[slice_idx_rel_to_batch] for d in pred[batch_idx]["detail"]],
                    dataset.init_res_level,
                    figsize,
                )

                fig_dwt.savefig(
                    f"{out_dir_coeffs / str(slice_idx)}.pdf",
                    dpi=dpi,
                    bbox_inches="tight",
                )
                plt.close(fig_dwt)


def export_wavelets(
    cfg: DictConfig,
    model: ContourModel,
    init_epoch: int,
    final_epoch: int,
    out_dir: Path,
    num_levels: int = 5,
    figsize: Tuple[int, int] = (20, 8),
    dpi: int = 100,
) -> None:
    """
    Export initial and final wavelet

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contours
    init_epoch: int
        initial epoch
    final_epoch: int
        final epoch
    out_dir: Path
        output path to store figures in
    num_levels: int
        number of levels used to sample wavelets
    figsize: Tuple[int, int]
        figure size
    dpi: int
        dots per inch
    """
    out_dir_wav = out_dir / "wavelets"
    out_dir_lpf = out_dir / "refinement_masks"
    for dir in (out_dir_wav, out_dir_lpf):
        if not dir.exists():
            dir.mkdir(parents=True)

    order_wavelet = cfg.network.decoder.order_wavelet

    for epoch in [init_epoch, final_epoch]:
        # Load low-pass filters
        model_utils.load_checkpoint(cfg, model, epoch=epoch)
        lpf = model.wavelet_layer.group_lpfs().squeeze(1)

        # Plot per spatial component (channel dimension)
        for comp_idx in range(lpf.shape[0]):
            lpf_comp = lpf[comp_idx].detach().cpu()

            # Plot wavelets
            fig_wavelet, ax_wavelet = plot_wavelets(
                lpf_comp, num_levels, figsize=figsize
            )
            filename = f"order_{order_wavelet}_epoch_{epoch}_comp_{comp_idx}"
            fig_wavelet.savefig(
                f"{out_dir_wav / filename}.pdf", dpi=dpi, bbox_inches="tight"
            )
            plt.close(fig_wavelet)

            # Plot refinement masks
            # fig_lpf, ax_lpf = plot_refinement_masks(lpf_comp, figsize=figsize)
            # fig_lpf.savefig(
            #     f"{out_dir_lpf / filename}.pdf", dpi=dpi, bbox_inches="tight"
            # )
            # plt.close(fig_lpf)
