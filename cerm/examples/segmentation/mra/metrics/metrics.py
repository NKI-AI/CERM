"""Module for constructing metrics to evaluate accuracy of predicted contours"""

import torch
import torch.fft as fft
import abc
import skimage.draw
import skimage.metrics
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
from shapely.geometry import Polygon

from mra.network.model import ContourModel
from mra.utils import fourier2d


def area_bbox(bbox: Tensor) -> Tensor:
    """
    Compute area of bounding box

    Parameters
    ----------
    bbox_gt: float-valued PyTorch tensor of shape [batch_size num_spatial_comp 2]
        corner points of bounding box

    Returns
    -------
    float-valued PyTorch tensor of shape [batch_size]
        area of each bounding box in batch
    """
    return (bbox[:, 0, 1] - bbox[:, 0, 0]) * (bbox[:, 1, 1] - bbox[:, 1, 0])


def generalized_intersection_over_union(bbox_gt: Tensor, bbox_pred: Tensor) -> Tensor:
    """
    Compute generalized intersection over union

    Parameters
    ----------
    bbox_gt: float-valued PyTorch tensor of shape [batch_size num_spatial_comp 2]
        corner points groundtruth bounding box
    bbox_pred: float-valued PyTorch tensor of shape [batch_size num_spatial_comp 2]
        corner points predicted bounding box

    Returns
    -------
    giou: float-valued PyTorch tensor of shape [batch_size]
        generalized intersection over union per sample
    """
    batch_size = bbox_gt.shape[0]

    # Determine potential corner points of intersection
    y_min = torch.max(bbox_gt[:, 0, 0], bbox_pred[:, 0, 0])
    y_max = torch.min(bbox_gt[:, 0, 1], bbox_pred[:, 0, 1])
    x_min = torch.max(bbox_gt[:, 1, 0], bbox_pred[:, 1, 0])
    x_max = torch.min(bbox_gt[:, 1, 1], bbox_pred[:, 1, 1])

    # Compute area's
    non_empty_idx = torch.where((y_max > y_min) & (x_max > x_min))[0]
    area_intersection = torch.zeros(batch_size).to(bbox_gt.device)
    if len(non_empty_idx) > 0:
        area_intersection[non_empty_idx] = (
            y_max[non_empty_idx] - y_min[non_empty_idx]
        ) * (x_max[non_empty_idx] - x_min[non_empty_idx])
    area_union = area_bbox(bbox_gt) + area_bbox(bbox_pred) - area_intersection

    # Check intersection is non-empty and compute intersection over union
    iou = area_intersection / area_union

    # Corner points convex hull bounding boxes
    cy_min = torch.min(bbox_gt[:, 0, 0], bbox_pred[:, 0, 0])
    cy_max = torch.max(bbox_gt[:, 0, 1], bbox_pred[:, 0, 1])
    cx_min = torch.min(bbox_gt[:, 1, 0], bbox_pred[:, 1, 0])
    cx_max = torch.max(bbox_gt[:, 1, 1], bbox_pred[:, 1, 1])
    area_convex_hull = (cy_max - cy_min) * (cx_max - cx_min)

    giou = iou - (area_convex_hull - area_union) / (area_convex_hull)

    return torch.mean(iou), torch.mean(giou)


def dice_mask(
    batch_contour_gt: List[Tensor],
    batch_contour_pred: Tensor,
    dim_img: Tuple[int, int],
    device: str = "cpu",
    eps: float = 1e-05,
) -> Tensor:
    """
    Compute dice score in naive manner.

    Parameters
    ----------
    batch_contour_gt: List[Tensor]
        batch of groundtruth curves stored in list
    batch_contour_pred: float-valued PyTorch tensor of size [batch_size num_comp num_pts]
        batch of predicted curves stacked in PyTorch tensor

    Returns
    -------
    Dice score
    """
    batch_size = len(batch_contour_gt)
    dice = torch.zeros(batch_size)

    for batch_idx, (contour_gt, contour_pred) in enumerate(
        zip(batch_contour_gt, batch_contour_pred)
    ):
        # Observed mask
        mask_gt = np.zeros(dim_img)
        rr, cc = skimage.draw.polygon(
            contour_gt[0, :].cpu(), contour_gt[1, :].cpu(), shape=dim_img
        )
        mask_gt[rr, cc] = 1

        # Predicted mask
        mask_pred = np.zeros(dim_img)
        rr, cc = skimage.draw.polygon(
            contour_pred[0, :].cpu(), contour_pred[1, :].cpu(), shape=dim_img
        )
        mask_pred[rr, cc] = 1

        dice[batch_idx] = (
            2
            * np.sum(mask_pred * mask_gt)
            / (np.sum(mask_pred) + np.sum(mask_gt) + eps)
        )

    # Dice
    return dice


def pointwise_distance_contours(contour_1: Tensor, contour_2: Tensor) -> Tensor:
    """
    Compute pointwise Euclidian distance between discretized contours

    Parameters
    ----------
    contour_1: float-valued PyTorch tensor of shape [batch_size num_spatial_comp num_points]
        groundtruth or predicted contour
    contour_2: float-valued PyTorch tensor of shape [batch_size num_spatial_comp num_points]
        groundtruth or predicted contour

    Returns
    -------
    distances: float-valued PyTorch tensor of size [batch_size num_pts_1 num_pts_2]
        pointwise Euclidian distance between contours
    """
    # Dimensions
    batch_size = contour_1.shape[0]
    num_pts_1 = contour_1.shape[-1]
    num_pts_2 = contour_2.shape[-1]

    # Preallocation
    distances = torch.zeros(batch_size, num_pts_1, num_pts_2).to(contour_1.device)

    # Compute ||gamma_1(p_i) - gamma_2(p_j)||
    for point_idx in range(num_pts_1):
        point_1 = contour_1[..., point_idx]
        diff = contour_2 - point_1.unsqueeze(-1).repeat(1, 1, num_pts_2)
        distances[:, point_idx] = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

    return distances


def avg_pointwise_distance_contours(contour_1: Tensor, contour_2: Tensor) -> Tensor:
    """
    Compute mean pointwise distance contours

    Parameters
    ----------
    contour_1: float-valued PyTorch tensor of shape [batch_size num_spatial_comp num_points]
        groundtruth or predicted contour
    contour_2: float-valued PyTorch tensor of shape [batch_size num_spatial_comp num_points]
        groundtruth or predicted contour

    Returns
    -------
    float-valued PyTorch tensor of shape [batch_size]
    """
    num_pts_1 = contour_1.shape[-1]
    num_pts_2 = contour_2.shape[-1]

    # Distance between points curves
    distances = pointwise_distance_contours(contour_1, contour_2)

    # Distances min_{y in c2} ||x - y|| for each x in contour 1
    distances_c1_c2 = torch.min(distances, 2)[0]

    # Distances min_{x in c1} ||y - x|| for each y in contour 2
    distances_c2_c1 = torch.min(distances, 1)[0]

    return (
        torch.sum(distances_c1_c2, dim=1) / num_pts_1
        + torch.sum(distances_c2_c1, dim=1) / num_pts_2
    )


def hausdorff(
    contour_1: Tensor, contour_2: Tensor, alpha: float = 0.95, reduce: bool = False
) -> Tensor:
    """
    Compute alpha-percentile hausdorff distance between curves

    Parameters
    ----------
    contour_1: float-valued PyTorch tensor of shape [batch_size num_spatial_comp num_points]
        groundtruth or predicted contour
    contour_2: float-valued PyTorch tensor of shape [batch_size num_spatial_comp num_points]
        groundtruth or predicted contour

    Returns
    -------
    hausdorff_scores: float-valued PyTorch tensor of shape [batch_size]
        alpha-percentile haussdorf distance between contours
    """
    # Distance between points curves
    distances = pointwise_distance_contours(contour_1, contour_2)

    # Distances min_{y in c2} ||x - y|| for each x in contour 1
    dist_c1_c2 = torch.min(distances, 2)[0]

    # Distances min_{x in c1} ||y - x|| for each y in contour 2
    dist_c2_c1 = torch.min(distances, 1)[0]

    # Need to loop over batch to discard outliers
    hausdorff_scores = []
    for dist_c1_c2_sample, dist_c2_c1_sample in zip(dist_c1_c2, dist_c2_c1):
        # sup_{x in c1} inf_{y in c2} || x - y ||
        q_alpha_12 = torch.quantile(dist_c1_c2_sample, alpha)
        sup_inf_12 = torch.max(dist_c1_c2_sample[dist_c1_c2_sample <= q_alpha_12])

        # sup_{y in c2} inf_{x in c1} || y - x ||
        q_alpha_21 = torch.quantile(dist_c2_c1_sample, alpha)
        sup_inf_21 = torch.max(dist_c2_c1_sample[dist_c2_c1_sample <= q_alpha_21])

        hausdorff_scores.append(torch.max(sup_inf_12, sup_inf_21))

    if reduce:
        return torch.mean(torch.tensor(hausdorff_scores))
    else:
        return torch.tensor(hausdorff_scores)


def close_curve(curve_component: List) -> List:
    return curve_component + [curve_component[0]]


def contour_to_polygon(curve: Tensor) -> Polygon:
    """
    Use wavelet coefficients to construct polygonal representation

    Parameters
    ----------
    curve: float-valued PyTorch tensor of size [num_comp num_pts]
    """
    return Polygon(
        zip(close_curve(curve[0].tolist()), close_curve(curve[1].tolist()))
    ).buffer(0)


def discrepancy_contours(
    batch_contour_gt: List[Tensor],
    batch_contour_pred: Tensor,
    device: str = "cpu",
    eps: float = 1e-05,
) -> Tuple[Tensor, Tensor]:
    """
    Compute dice and haussdorf score using contour points.

    Parameters
    ----------
    batch_contour_gt: List[Tensor]
        batch of groundtruth curves stored in list
    batch_contour_pred: float-valued PyTorch tensor of size [batch_size num_comp num_pts]
        batch of predicted curves stacked in PyTorch tensor

    Returns
    -------
    Tuple[Tensor, Tensor]
        dice and hausdorff scores for each sample in batch
    """
    batch_size = len(batch_contour_gt)
    dice = torch.zeros(batch_size)
    hausdorff = torch.zeros(batch_size)

    for batch_idx, (contour_gt, contour_pred) in enumerate(
        zip(batch_contour_gt, batch_contour_pred)
    ):
        c1 = contour_to_polygon(contour_gt)
        c2 = contour_to_polygon(contour_pred)

        if c1.intersects(c2):
            c1_cap_c2_area = c1.intersection(c2).area
        else:
            c1_cap_c2_area = 0

        dice[batch_idx] = (2 * c1_cap_c2_area) / (c1.area + c2.area + eps)
        hausdorff[batch_idx] = c1.hausdorff_distance(c2)

    return dice, hausdorff


def total_variation_intensity(batch_img: Tensor, batch_contour: Tensor) -> Tensor:
    """
    Integrate L2-norm image gradient around contour

    Parameters
    ----------
    img_img: float-valued PyTorch tensor of shape [batch_size height width]
        batch of images
    batch_contour: float-valued PyTorch tensor of size [batch_size 2 num_pts]
        batch of contours sampled on uniformly spaced grid of [-1/2, 1/2]

    Returns
    -------
    l2_norm_img_grad: float-valued PyTorch tensor of shape [batch_size]
        l2 norm image gradient integrated along contours
    """
    # Fourier expansion image
    period_img = torch.tensor(batch_img.shape[1:]).to(batch_img.device) - 1
    batch_img_fourier_coeffs = fourier2d.fourier_coeffs(batch_img)

    # Order wavelet decomposition
    num_pts = batch_contour.shape[-1]
    assert num_pts % 2 == 0, "Number of samples points is not even!"
    order_curve = num_pts // 2

    # ...
    l2_norm_img_grad = []

    for img_fourier_coeffs, contour_pts in zip(batch_img_fourier_coeffs, batch_contour):
        # Evaluate grad(I) along contour
        img_fourier_series = fourier2d.FourierSeries2d(
            img_fourier_coeffs.unsqueeze(0), period_img
        )
        img_grad = img_fourier_series.diff()
        img_grad_eval = img_grad(torch.transpose(contour_pts, 0, 1))

        # Evaluate || grad I ||^2_{2} along contour and Fourier coefficients
        squared_norm_img_grad = torch.sum(img_grad_eval**2, dim=0)
        l2_norm_img_grad.append(
            torch.real(fft.fft(squared_norm_img_grad)[0]) / (2 * order_curve)
        )

    return torch.stack(l2_norm_img_grad)
