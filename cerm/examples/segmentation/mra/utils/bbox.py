"""Utilities for learning bounding boxes during training"""

import torch
import numpy as np

from torch import Tensor
from typing import Tuple

from mra.utils.fourier2d import FourierSeries2d


def scale_bbox_param(param: Tensor, scale_factor: Tensor) -> Tensor:
    """
    Scale bounding box parameters: midpoint or heigth and width

    Parameters
    ----------
    param: float-valued PyTorch tensor of size [batch_size num_spatial_dim]
        bounding box parameter: midpoint or height and width
    scale_factor: float-valued PyTorch tensor of size [2]
        scaling factor for heigth, width and midpoint

    Returns
    -------
    scaled_params: float-valued PyTorch tensor of size [batch_size num_spatial_dim]
        rescaled parameter
    """
    return torch.einsum("ij, j -> ij", param, scale_factor)


def midrad_bbox_from_contour(contour: Tensor, bbox_margin: int = 0) -> Tensor:
    """
    Construct bounding box around contour using midpoint-radius representation

    Parameters
    ----------
    contour: float-valued PyTorch tensor of size [num_spatial_dim num_pts]
        discretization of closed curve
    margin: int
        length added to measurements rectangle to allow for some wiggle room

    Returns
    -------
    midpoint: float-valued PyTorch tensor of size [num_spatial_dim]
        midpoint of bounding box
    heigth_width: float-valued PyTorch tensor of size [num_spatial_dim]
        height and width of bounding box
    """
    num_spatial_dim = contour.shape[0]
    min_points = [
        torch.min(contour[spatial_idx]) for spatial_idx in range(num_spatial_dim)
    ]
    max_points = [
        torch.max(contour[spatial_idx]) for spatial_idx in range(num_spatial_dim)
    ]

    height_width = torch.tensor(
        [
            (max_points[spatial_idx] - min_points[spatial_idx]) + bbox_margin
            for spatial_idx in range(num_spatial_dim)
        ]
    )

    midpoint = torch.tensor(
        [
            (max_points[spatial_idx] + min_points[spatial_idx]) / 2
            for spatial_idx in range(num_spatial_dim)
        ]
    )

    return midpoint, height_width


def midrad_to_coords_bbox(
    midpoint: Tensor,
    height_width: Tensor,
    scale_factor: Tensor,
    num_spatial_dim: int = 2,
) -> Tensor:
    """
    Compute corner points bounding box using midpoint-radius representation

    Parameters
    ----------
    midpoint: float-valued PyTorch tensor of size [batch_size num_spatial_dim]
        midpoint of bounding box
    height_width: float-valued PyTorch tensor of size [batch_size num_spatial_dim]
        height and width of bounding box
    scale_factor: float-valued PyTorch tensor of size [num_spatial_dim]
        scaling factor for heigth, width and midpoint
    """
    midpoint_scaled = scale_bbox_param(midpoint, scale_factor)
    height_width_scaled = scale_bbox_param(height_width, scale_factor)
    return torch.stack(
        (
            midpoint_scaled - height_width_scaled / 2,
            midpoint_scaled + height_width_scaled / 2,
        ),
        dim=-1,
    )


def coords_bbox_from_contour(contour: Tensor, eps: float = 10) -> Tensor:
    """
    Construct bounding box contour using corner-point representation

    Parameters
    ----------
    contour: float-valued PyTorch tensor of size [num_spatial_dim num_pts]
        discretization of closed curve
    eps: int
        length added to measurements rectangle to allow for some wiggle room

    Returns
    -------
    coords: float-valued PyTorch tensor of shape [num_spatial_dim num_spatial_dim]
        coordinates of corner points
    """
    num_spatial_dim = contour.shape[0]
    coords_bbox = torch.zeros(num_spatial_dim, num_spatial_dim)

    # Lower bounds coordinates per spatial dimension
    coords_bbox[:, 0] = (
        torch.tensor(
            [torch.min(contour[spatial_idx]) for spatial_idx in range(num_spatial_dim)]
        )
        - eps
    )

    # Upper bounds coordinates per spatial dimension
    coords_bbox[:, 1] = (
        torch.tensor(
            [torch.max(contour[spatial_idx]) for spatial_idx in range(num_spatial_dim)]
        )
        + eps
    )

    return coords_bbox


class BumpFunction:
    """2-dimensional bump function with rectangulr support"""

    def __init__(
        self, dim_img: Tuple[int, int], order: Tuple[int, int], sigma: Tensor
    ) -> None:
        """
        Initialize parameters fourier approximation bump function

        Parameters
        ----------
        dim_img: Tuple[int, int]
            dimension of 2d image [heigth width]
        order: Tuple[int, int]
            order of Fourier approximation in each direction
        sigma: float-valued PyTorch tensor of size [2]
            size of region used to average Fourier series
        """
        # Dimensions
        self.dim_img = dim_img
        self.num_spatial_dim = len(dim_img)
        self.order = order

        # Parameters Fourier approximation
        self.period = torch.tensor(
            [dim_img[idx] - 1 for idx in range(self.num_spatial_dim)]
        )
        self.angular_freq = 2 * np.pi / self.period

        # Precompute mode indices excluding zeroth index
        self.modes = [
            torch.cat(
                (
                    torch.arange(1 - self.order[spatial_idx], 0),
                    torch.arange(1, self.order[spatial_idx]),
                )
            )
            .unsqueeze(0)
            .float()
            for spatial_idx in range(self.num_spatial_dim)
        ]

        # Indices nonzero modes
        self.nonzero_idx = [
            list(range(self.order[spatial_idx] - 1))
            + list(range(self.order[spatial_idx], 2 * self.order[spatial_idx] - 1))
            for spatial_idx in range(self.num_spatial_dim)
        ]

        # Lancosz smoothing factor
        self.sigma = sigma
        lancosz_factors = [
            torch.sinc(
                torch.arange(1 - self.order[spatial_idx], self.order[spatial_idx])
                * self.sigma[spatial_idx]
                / self.period[spatial_idx]
            )
            for spatial_idx in range(self.num_spatial_dim)
        ]
        self.lancosz_factor = lancosz_factors[0].unsqueeze(1) @ lancosz_factors[
            1
        ].unsqueeze(0)

        # Store grid
        x = torch.arange(dim_img[1]).unsqueeze(0).repeat(dim_img[0], 1)
        y = torch.arange(dim_img[0]).unsqueeze(1).repeat(1, dim_img[1])
        self.grid = torch.stack((y.flatten(), x.flatten()), dim=1).float()

    def fourier_coeffs(self, coords_rectangle: Tensor) -> Tensor:
        """
        Fourier coefficients of indicator function rectangle

        Parameters
        ----------
        coords_rectangle: float-valued PyTorch Tensor of shape [batch_size 2 2]
            coordinates corner-points rectangle
        device: str
            device on which computations are performed
        """
        # Initialize arrays on device
        device = coords_rectangle.device
        batch_size = coords_rectangle.shape[0]
        angular_freq = self.angular_freq.to(device)
        period = self.period.to(device)
        modes = [
            self.modes[spatial_idx].to(device)
            for spatial_idx in range(self.num_spatial_dim)
        ]
        coeffs = torch.zeros(
            batch_size,
            2 * self.order[0] - 1,
            2 * self.order[1] - 1,
            dtype=torch.complex64,
        ).to(device)

        # Compute length of intervals making up rectangle
        rectangle_measures = [
            coords_rectangle[:, spatial_idx, 1] - coords_rectangle[:, spatial_idx, 0]
            for spatial_idx in range(self.num_spatial_dim)
        ]

        # Construct difference exponentials in each spatial direction
        exp_diff_comp = []
        for spatial_idx in range(self.num_spatial_dim):
            exp_diff_comp.append(
                torch.exp(
                    -1j
                    * angular_freq[spatial_idx]
                    * (
                        coords_rectangle[:, spatial_idx, 1].unsqueeze(1)
                        @ modes[spatial_idx]
                    )
                )
                - torch.exp(
                    -1j
                    * angular_freq[spatial_idx]
                    * (
                        coords_rectangle[:, spatial_idx, 0].unsqueeze(1)
                        @ modes[spatial_idx]
                    )
                )
            )

        # Compute Fourier coefficients

        # a(k1, k2): both k1 and k2 nonzero
        exp_eval = torch.einsum("ik, il -> ikl", exp_diff_comp[0], exp_diff_comp[1])
        a = -exp_eval / (4 * np.pi**2 * torch.transpose(modes[0], 0, 1) @ modes[1])
        coeffs[:, 0 : self.order[0] - 1, 0 : self.order[1] - 1] = a[
            :, 0 : self.order[0] - 1, 0 : self.order[1] - 1
        ]
        coeffs[:, 0 : self.order[0] - 1, self.order[1] :] = a[
            :, 0 : self.order[0] - 1, self.order[1] - 1 :
        ]
        coeffs[:, self.order[0] :, 0 : self.order[1] - 1] = a[
            :, self.order[0] - 1 :, 0 : self.order[1] - 1
        ]
        coeffs[:, self.order[0] :, self.order[1] :] = a[
            :, self.order[0] - 1 :, self.order[1] - 1 :
        ]

        # Precompute factors for a(k1, k2), where k1 or k2 is zero
        factors = [
            1j
            * rectangle_measures[(spatial_idx + 1) % 2]
            / (2 * np.pi * period[(spatial_idx + 1) % 2])
            for spatial_idx in range(self.num_spatial_dim)
        ]

        # a(k1, 0): k1 nonzero
        coeffs[:, self.nonzero_idx[0], self.order[1] - 1] = (
            factors[0].unsqueeze(1)
            @ (1 / modes[0]).type(torch.complex64)
            * exp_diff_comp[0]
        )

        # a(0, k2): k2 nonzero
        coeffs[:, self.order[0] - 1, self.nonzero_idx[1]] = (
            factors[1].unsqueeze(1)
            @ (1 / modes[1]).type(torch.complex64)
            * exp_diff_comp[1]
        )

        # a(0, 0)
        coeffs[:, self.order[0] - 1, self.order[1] - 1] = (
            rectangle_measures[0] * rectangle_measures[1] / torch.prod(period)
        )

        return coeffs

    def __call__(self, coords_rectangle: Tensor, points: Tensor) -> Tensor:
        device = coords_rectangle.device
        smoothed_coeffs = torch.einsum(
            "ijk, jk -> ijk",
            self.fourier_coeffs(coords_rectangle),
            self.lancosz_factor.to(device),
        )
        bump = FourierSeries2d(smoothed_coeffs, self.period)
        return bump(points)
