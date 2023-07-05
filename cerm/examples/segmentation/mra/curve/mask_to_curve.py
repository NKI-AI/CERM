"""Module for converting binary mask (slice) to contour."""

import numpy as np
import torch
import torch.fft as fft
import warnings
import cv2

from mra.curve import fourier
from mra.utils import bbox

from typing import Tuple, List
from skimage import morphology
from torch import Tensor


def machine_eps(a: Tensor, eps: float = 1e-02) -> Tensor:
    """
    Set (Fourier) coefficients below threshold to zero.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (..., order)
        array representing Fourier coefficients

    Returns
    -------
    a: complex-valued Pytorch tensor of size (..., order)
        Fourier coefficients with elements below eps set to zero
    """
    a = torch.view_as_real(a)
    a[torch.abs(a) < eps] = 0
    return torch.view_as_complex(a)


@torch.jit.script
def fourier_coeffs(curve: Tensor, order: int) -> Tuple[Tensor, Tensor]:
    """
    Compute Fourier coefficients of closed curve parameterized by arclength.

    Parameters
    ----------
    curve: float-valued numpy array of size (num_components, num_pts)
        points on discretized curve
    order: int
        order of Fourier expansion

    Returns
    -------
    float
        length of curve
    dft_curve: complex-valued numpy array of size (num_components, order)
        Fourier coefficients of the curve
    """
    # Ensure that curve is closed
    curve[:, -1] = curve[:, 0]

    # Compute arclength length curve
    num_pts = curve.shape[1]
    arclen = torch.zeros(num_pts)
    arclen[1:] = torch.cumsum(
        torch.sqrt(torch.sum((curve[:, 1:] - curve[:, 0 : num_pts - 1]) ** 2, dim=0)),
        dim=0,
    )

    # Sample on equispaced grid of [0, length_curve] excluding endpoint
    num_sample_pts = 2 * order - 1
    grid = torch.linspace(0, arclen[-1], num_sample_pts + 1)[0:num_sample_pts]
    dft_curve = torch.zeros((curve.shape[0], num_sample_pts))
    rhs_idx = 0
    for idx_t, t in enumerate(grid):
        # Locate right endpoint of interval in which t is located
        while t >= arclen[rhs_idx]:
            rhs_idx += 1
        # Use linear interpolation to evaluate curve at t
        dft_curve[:, idx_t] = (curve[:, rhs_idx] - curve[:, rhs_idx - 1]) / (
            arclen[rhs_idx] - arclen[rhs_idx - 1]
        ) * (t - arclen[rhs_idx - 1]) + curve[:, rhs_idx - 1]

    coeffs = torch.conj(fft.ifft(dft_curve, dim=1)[:, 0:order])

    return arclen[-1], machine_eps(coeffs)


def shift_idx(start_idx: int, num_pts: int) -> Tensor:
    """
    Shift curve so that it starts at start_idx.

    Parameters:
    -----------
    start_idx: int
        start index where curve should start
    num_pts: int
        number of points on discretized curve

    Returns
    -------
    int-valued PyTorch tensor of size [num_pts]
        shifted indices
    """
    return np.mod(np.arange(start_idx, start_idx + num_pts), num_pts)


def polar_repr(
    curve: Tensor, midpt: Tensor, num_comp: int = 2, eps: float = 1.5
) -> Tensor:
    """
    Compute polar representation of curve.

    Parameters
    ----------
    curve: float-valued PyTorch tensor of size [num_comp num_pts]
        discretization of closed curve
    midpt: float-valued PyTorch tensor of size [num_comp]
        geometric center of curve

    Returns
    -------
    polar: float-valued PyTorch tensor of size [2 num_pts]
        polar representation of curve with respect to midpoint
    """
    # Compute polar coordinates
    polar = torch.zeros(num_comp, curve.shape[1])
    curve_centered = curve - midpt.view(num_comp, 1)
    radius = torch.sqrt(torch.sum(curve_centered**2, dim=0))
    angle = torch.acos(curve_centered[1] / radius)
    y_neg_idx = torch.where(curve_centered[0] < 0)[0]
    angle[y_neg_idx] = 2 * np.pi - angle[y_neg_idx]

    # Compute Fourier coefficients of angle and radius
    start_idx = torch.argmin(angle)
    idx = shift_idx(start_idx + 1, curve.shape[1])
    angle = angle[idx]
    radius = radius[idx]

    # return curve[:, idx], torch.vstack((radius, angle))
    return curve[:, idx]


def contour_from_mask(
    mask: np.ndarray,
    init_res_level: int,
    order_fourier: int,
    mean_midpt: Tensor = torch.zeros(2, 1),
    min_num_pix: int = 5,
    bbox_margin: int = 2,
    num_comp: int = 2,
    min_dist: float = 2,
    threshold: float = 1e-03,
    min_obj_size: int = 5,
    num_con_comp: int = 1,
) -> Tensor:
    """
    Extract boundary from binary mask and compute its wavelets coefficients.

    Parameters
    ----------
    mask: binary-valued numpy array of size (height, width)
        binary mask (slice)
    order_fourier: int
        order of Fourier expansion
    min_num_pix: int, optional
        minimal number of pixels a slice should contain for it to be considered non-empty
    bbox_margin: int
        number of pixels added to height and width of bounding box
    num_comp: int, optional
        number of phase space components
    min_dist: float, optional
        minimal distance between start and endpoint of curve

    Returns
    -------
    arclen: float-valued numpy array of size (num_slices, num_con_comp)
        arclength of contour
    contour: float-valued PyTorch array of size (num_comp, num_sample_points)
        discretized contour extracted from mask using opencv
    contour_dyadic_scaled:
        discretized contour on dyadic grid obtained using Fourier series
    """
    is_corrupt = False
    is_empty = False
    dim_img = mask.shape
    size_dyadic_grid = 2**init_res_level
    period = 1  # Curves are re-parameterized to have period 1
    dyadic_grid = (
        torch.arange(-(2 ** (init_res_level - 1)), 2 ** (init_res_level - 1)).float()
        / size_dyadic_grid
    )

    # Preallocation
    arclen = torch.tensor(0.0)
    midpoint_contour = torch.zeros(num_comp)
    midpoint_bbox = torch.zeros(num_comp)
    height_width_bbox = torch.zeros(num_comp)
    contour_dyadic_scaled = torch.zeros(num_comp, size_dyadic_grid)
    contour_opencv = torch.zeros(num_comp, size_dyadic_grid)

    # Check if sample is empty
    if torch.sum(mask) < min_num_pix:
        is_empty = True
        return (
            is_empty,
            is_corrupt,
            arclen,
            contour_opencv,
            contour_dyadic_scaled,
            midpoint_contour,
            midpoint_bbox,
            height_width_bbox,
        )

    # Clean mask and find contours
    mask_clean = morphology.remove_small_objects(
        np.array(mask.numpy(), dtype=bool), min_obj_size
    )
    contour_list, hierarchy = cv2.findContours(
        mask_clean.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    if len(contour_list) > num_con_comp:
        warnings.warn("There can be at most one contour")
        is_corrupt = True

    elif len(contour_list) == num_con_comp:
        contour = np.squeeze(contour_list[0], axis=1)

        if contour.shape[0] < min_num_pix:
            warnings.warn("Skipping contour: number of pixels in mask is too small")
            is_corrupt = True
        elif np.linalg.norm(contour[-1, :] - contour[0, :], ord=2) > min_dist:
            warnings.warn(f"Skip: prescribed curve is not closed; distance")
            is_corrupt = True
        else:
            # Store "original" contour
            contour_opencv = torch.from_numpy(
                np.flip(np.transpose(contour), axis=0).copy()
            )

            # Parameterize contour by arclength and compute Fourier coefficients
            arclen, coeffs = fourier_coeffs(contour_opencv, order_fourier)

            # Evaluate contour on dyadic grid
            contour_dyadic = fourier.eval(
                coeffs, float(arclen), float(arclen) * dyadic_grid
            )

            # Orient contour and scale to get approximation coefficients
            midpoint_contour = fourier.midpoint(
                coeffs.unsqueeze(0).unsqueeze(0),
                arclen.unsqueeze(0).unsqueeze(0),
            )[0, 0]
            contour_dyadic_scaled = 2 ** (-init_res_level / 2) * (
                polar_repr(contour_dyadic, midpoint_contour) - mean_midpt
            )

            # Compute bounding box
            midpoint_bbox, height_width_bbox = bbox.midrad_bbox_from_contour(
                contour_dyadic, bbox_margin=bbox_margin
            )

    return (
        is_empty,
        is_corrupt,
        arclen,
        contour_opencv,
        contour_dyadic_scaled,
        midpoint_contour,
        midpoint_bbox,
        height_width_bbox,
    )
