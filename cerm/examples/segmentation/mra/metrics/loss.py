"""Module for constructing metrics to evaluate accuracy of predicted contours"""

import torch
import torch.fft as fft
import abc
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Union

from mra.network.model import ContourModel
from mra.utils import inference
from mra.metrics import metrics


class ContourLoss:
    """Bbox loss + contour loss + cross-entropy object present or not"""

    def __init__(
        self, weights: Dict[str, float], mean_midpoint: Tensor, init_res_level: int
    ) -> None:
        """
        Initialize name and type of metric.

        Parameters
        ----------
        weights: Dict[str, float]
            weights for low resolution, high resolution and cross-entropy terms
        """
        self.name = "Contour-loss"
        self.weights = weights
        self.mean_midpoint = mean_midpoint
        self.init_res_level = init_res_level

    def __call__(
        self,
        sample: Dict[str, Union[Tensor, List[Tensor]]],
        pred: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
    ) -> Tensor:
        """Compute classification, low resolution and high resolution losses."""
        device = sample["is_non_empty"].device
        dim_img = sample["img"].shape[-2:]
        loss = {}

        # Contour and bounding box loss
        non_void_idx = torch.where(sample["is_non_empty"] == 1)[0]

        if len(non_void_idx) > 0:
            # Contour: L2-error approximation coefficients highest level
            scale_factor = 2 ** (self.init_res_level / 2)
            approx_high_obs = sample["contour_dyadic_scaled"][non_void_idx]
            approx_high_pred = pred["approx"][-1][non_void_idx]
            loss["approx_high_l2"] = l2_norm(approx_high_obs, approx_high_pred)

            # Active contour loss
            if self.weights["active_contour"] > 0:
                img = sample["img"].squeeze(1)[non_void_idx]
                contour_pred = scale_factor * approx_high_pred + self.mean_midpoint.to(
                    device
                )
                loss["active_contour"] = -torch.mean(
                    torch.sqrt(metrics.total_variation_intensity(img, contour_pred))
                )
            else:
                loss["active_contour"] = 0.0

        # Complete loss
        complete_loss = torch.tensor(0.0, requires_grad=True).to(device)
        for loss_key, loss_weight in self.weights.items():
            complete_loss = complete_loss + loss_weight * loss[loss_key]
        loss["complete_loss"] = complete_loss

        return loss


def mean_loss(
    model: ContourModel,
    loss: ContourLoss,
    data_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Compute mean (empirical average) loss and metrics.

    Parameters
    ----------
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contours
    loss: ContourLoss
        loss function
    data_loader: DataLoader
        set on which the average loss is computed
    device: str in {'cpu', 'gpu'}
        device used to perform computations

    Returns
    -------
    mean_loss: Dict[str, Tensor]
        average of components and complete loss over dataset
    """
    # Initialization
    dim_img = data_loader.dataset.dataset.dim_img
    init_res_level = data_loader.dataset.dataset.init_res_level
    mean_midpoint = data_loader.dataset.dataset.mean_midpoint
    batch_size = data_loader.batch_size
    mean_loss = {}

    # Set up loss keys which are only averaged over non-empty slices
    performance_keys = [
        "approx_high_l2",
        "dice",
        "hausdorff",
        "hausdorff95",
        "active_contour",
    ]

    for loss_key in performance_keys:
        mean_loss[loss_key] = torch.tensor(0.0)

    with torch.no_grad():
        # Perform inference over entire dataset first to keep GPU busy
        obs, pred = inference.predict(model, data_loader, device)

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

            # Evaluate loss
            loss_eval = loss(obs_batch, pred_batch)

            # Average contour loss and metrics over non-empty slices only
            for key in ["approx_high_l2", "active_contour"]:
                mean_loss[key] += num_non_empty_in_batch * loss_eval[key]

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
            mean_loss["dice"] += torch.sum(dice)
            mean_loss["hausdorff"] += torch.sum(hausdorff)
            mean_loss["hausdorff95"] += torch.sum(
                metrics.hausdorff(contour_gt, contour_pred)
            )

        # Average each component in loss
        for loss_key in performance_keys:
            mean_loss[loss_key] /= num_non_empty_samples

        # Average complete loss
        complete_loss = torch.tensor(0.0)
        for loss_key, loss_weight in loss.weights.items():
            complete_loss = complete_loss + loss_weight * mean_loss[loss_key]
        mean_loss["complete_loss"] = complete_loss

    return mean_loss


def l2_norm(x: Tensor, y: Tensor) -> Tensor:
    return torch.mean(torch.sum(torch.sqrt(torch.sum((x - y) ** 2, dim=-1)), dim=-1))


def l1_norm(x: Tensor) -> Tensor:
    return torch.mean(torch.sum(torch.abs(x), dim=[1, 2]))


def logcosh(x: Tensor, y: Tensor) -> Tensor:
    s = x - y
    val = s + torch.log(1 + torch.exp(-2 * s)) - torch.log(torch.tensor(2.0))
    return torch.mean(torch.sum(val, dim=[1, 2]))


def l2_norm_grad_loss(model: ContourModel) -> float:
    """
    Compute ||D_{w}E(L(x, y, w))||_2 for L: X x Y -> [0, infty) loss function.

    Parameters
    ----------
    model: ContourModel (inherited class of nn.Module)
        neural network modelling Fourier coefficients of contours

    Returns
    -------
    float
        ||D_{w}E(L(x, y, w))||_2
    """
    y = torch.tensor([0]).float()
    with torch.no_grad():
        for weights in model.parameters():
            if weights.grad is not None:  # There is only one None object so use "is"
                y += torch.sum(weights.grad**2).cpu()
    return float(torch.sqrt(y))
