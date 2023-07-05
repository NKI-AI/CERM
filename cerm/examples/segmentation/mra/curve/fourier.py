"""Module for performing computations with Fourier series."""

import numpy as np
import torch
import warnings
import cv2
import torch.fft as fft

from torch import Tensor


def machine_eps(coeffs: Tensor, eps: float = 1e-03) -> Tensor:
    """
    Set Fourier coefficients below threshold to zero.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (..., order)
        array representing Fourier coefficients

    Returns
    -------
    coeffs: complex-valued Pytorch tensor of size (..., order)
        array representing Fourier coefficients where elements below eps are set to zero
    """
    coeffs[torch.where(torch.abs(coeffs) < eps)] = 0
    return coeffs


def truncate(coeffs: np.ndarray, num_comp: int = 2, eps: float = 0.15) -> np.ndarray:
    """
    Set higher order Fourier coefficients to zero by locating when L1-norm increases linearly.

    Parameters
    ----------
    coeffs: complex-valued numpy array of size (num_comp, order)
        Fourier coefficients
    num_comp: int
        number of components in phase space
    eps: float
        threshold used to determine linearity in least squares approximation

    Returns
    -------
    coeffs: complex-valued numpy array of size (num_comp, order)
        Truncated Fourier coefficients
    """
    order = coeffs.shape[-1]
    cumsum = np.cumsum(np.abs(coeffs), axis=-1)
    for comp_idx in range(num_comp):
        trunc_order = 0
        residual = 2 * eps
        while residual > eps:
            ind = np.reshape(np.arange(trunc_order, order), (order - trunc_order, 1))
            A = np.concatenate((ind, np.ones((order - trunc_order, 1))), axis=1)
            A_adj = np.transpose(A)
            line = np.linalg.solve(A_adj @ A, A_adj @ cumsum[comp_idx, trunc_order:])
            residual = np.linalg.norm(A @ line - cumsum[comp_idx, trunc_order:])
            trunc_order += 1
        coeffs[comp_idx, trunc_order:] = 0
    return coeffs


def pad(coeffs: Tensor, num_zeros: int) -> Tensor:
    """
    Pad Fourier coefficients with zeros.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (..., order)
        array representing Fourier coefficients
    num_zeros: number of zeros to padd with

    Returns
    -------
    complex-valued Pytorch tensor of size (..., order + num_zeros)
        two-sided convolution of a and b
    """
    return torch.nn.functional.pad(coeffs, (0, num_zeros))


def fft_shift(coeffs: Tensor) -> Tensor:
    """
    Map two-sided Fourier coefficients to equivalent one-sided form when evaluated on S1.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (..., order)
        one-sided representation Fourier coefficients

    Returns
    -------
    complex-valued Pytorch tensor of size (..., 2 * order - 1)
        one-sided representation of Fourier coefficients when evaluated on S1.
    """
    return torch.cat(
        (coeffs, torch.conj(torch.flip(coeffs[..., 1:], dims=[-1]))), dim=-1
    )


def conv1d(a: Tensor, b: Tensor, method="fft") -> Tensor:
    """
    Compute two-sided convolution a*b using fft.

    Parameters
    ----------
    a: complex-valued Pytorch tensor of size (...., order_a)
        array representing Fourier coefficients
    b: complex-valued Pytorch tensor of size (....., order_b)
        array representing Fourier coefficients
    method: str in {fft, sum}, optional
        method used for computing convolution

    Returns
    -------
    a_conv_b: complex-valued Pytorch tensor of size (order_a + order_b - 1, num_comps)
        two-sided convolution of a and b
    """
    # Initialization
    order_a = a.size()[-1]
    order_b = b.size()[-1]
    order_ab = order_a + order_b - 1

    # Compute convolution
    fft_a = fft.fft(fft_shift(pad(a, order_b - 1)), dim=-1)
    fft_b = fft.fft(fft_shift(pad(b, order_a - 1)), dim=-1)

    return machine_eps(fft.ifft(fft_a * fft_b, dim=-1)[..., 0:order_ab])


def zeroth_conv_term(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute zero-th term of convolution (a * b). If a and b are Fourier series of functions f
    and g, repectively, with the same period, then (a * b)_0 is the average of fg.

    Parameters
    ----------
    a: complex-valued Pytorch tensor of size (...., order_a)
        array representing Fourier coefficients
    b: complex-valued Pytorch tensor of size (....., order_b)
        array representing Fourier coefficients

    Returns
    -------
    complex-valued Pytorch tensor of size (...)
        (a * b)_0
    """
    min_order = min(a.size()[-1], b.size()[-1])
    return torch.real(a[..., 0] * b[..., 0]) + 2 * torch.sum(
        torch.real(a[..., 1:min_order] * torch.conj(b[..., 1:min_order])), dim=-1
    )


def integrate(coeffs: Tensor, period: Tensor, device: str = "cpu") -> Tensor:
    """
    Compute Fourier coefficients of integral t -> int_{0}^{t} f(s) ds. We assume that
    the zeroth order Fourier coefficient, i.e., the average of f, is zero.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation of Fourier coefficients
    period: float-valued Pytorch tensor of size(batch_size, num_classes)
        period associated to Fourier series
    device: str in {'cpu', 'gpu'}
        device used to perform computations

    Returns
    -------
    complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        Fourier coefficients of t -> int_{0}^{t} f(s) ds
    """
    # Indices Fourier coefficients
    shape = coeffs.size()
    indices = torch.arange(1, shape[-1]).view(
        [1 for idx in range(len(shape) - 1)] + [shape[-1] - 1]
    )
    indices = indices.to(device)

    # Fourier coefficients of integral
    integral = torch.zeros(shape, dtype=torch.cfloat).to(device)
    weight = -1j * period / (2 * np.pi)
    integral[..., 1:] = (
        torch.einsum("ijkl, ij -> ijkl", coeffs[..., 1:], weight) / indices
    )
    integral[..., 0] = 2 * torch.sum(torch.imag(-1j * integral[..., 1:]), dim=-1)

    return integral


def diff(
    coeffs: Tensor,
    period: Tensor,
    deriv_order: int = 1,
    device: str = "cpu",
    eps: float = 1e-09,
) -> Tensor:
    """
    First order derivative Fourier series.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation Fourier coefficients
    period: float
        period of Fourier series
    deriv_order: int, optional
        order of derivative
    device: str in {'cpu', 'gpu'}
        device used to perform computations
    eps: float, optional
        threshold used to determine whether period is zero

    Returns
    -------
    complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation of derivative Fourier series
    """
    # Initialization
    shape = coeffs.size()
    indices = torch.arange(0, shape[-1]).view(
        [1 for idx in range(len(shape) - 1)] + [shape[-1]]
    )
    indices = indices.to(device)

    # Inefficient manner to deal with zero period
    freq = torch.zeros(period.shape, dtype=period.dtype).to(device)
    nonzero_idx = torch.where(torch.abs(period) > eps)
    freq[nonzero_idx] = 2 * np.pi / period[nonzero_idx]

    # Precompute weights in order to avoid unnecessary computations with tensors
    if deriv_order == 1:
        weight = 1j * freq
    elif deriv_order == 2:
        weight = -(freq**2)
        indices = indices**2
    else:
        raise NotImplementedError(
            f"Differentiation not implemented for order {deriv_order}"
        )

    return indices * torch.einsum("ijkl, ij -> ijkl", coeffs, weight)


def energy(
    coeffs: Tensor, period: Tensor, deriv: Tensor = [], device: str = "cpu"
) -> Tensor:
    """
    Compute Fourier expansion of squared length velocity vector.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation Fourier coefficients
    period: float
        period of Fourier series
    device: str in {'cpu', 'gpu'}
        device used to perform computations

    Returns
    -------
    complex-valued PyTorch tensor of size (batch_size, num_classes, 2 * order - 1)
        Fourier coefficients of squared length velocity vector
    """
    if len(deriv) == 0:
        deriv = diff(coeffs, period, device=device, deriv_order=1)
    return conv1d(deriv[..., 0, :], deriv[..., 0, :]) + conv1d(
        deriv[..., 1, :], deriv[..., 1, :]
    )


def enclosed_area(
    coeffs: Tensor, period: Tensor, deriv: Tensor = [], device: str = "cpu"
) -> Tensor:
    """
    Compute area enclosed by contour.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation Fourier coefficients
    period: float-valued PyTorch tensor of size (batch_size, num_classes)
        period of Fourier series
    deriv: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation of derivative of Fourier expansion
    device: str in {'cpu', 'gpu'}
        device used to perform computations

    Returns
    -------
    float-valued PyTorch tensor of size (batch_size, num_classes)
        area enclosed by contours
    """
    if len(deriv) == 0:
        deriv = diff(coeffs, period, device=device, deriv_order=1)
    return torch.abs(period * zeroth_conv_term(coeffs[..., 0, :], deriv[..., 1, :]))


def midpoint(
    coeffs: Tensor,
    period: Tensor,
    deriv: Tensor = [],
    area: Tensor = [],
    device: str = "cpu",
) -> Tensor:
    """
    Compute center of 2d contour using Green's Theorem.

    Parameters
    ----------
    coeffs: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation Fourier coefficients
    period: float-valued PyTorch tensor of size (batch_size, num_classes)
        period of Fourier series
    deriv: complex-valued Pytorch tensor of size (batch_size, num_classes, num_comp, order)
        one-sided representation of derivative of Fourier expansion
    area: complex-valued Pytorch tensor of size (batch_size, num_classes)
        area enclosed by contour
    device: str in {'cpu', 'gpu'}
        device used to perform computations

    Returns
    -------
    area: float-valued PyTorch tensor of size (batch_size, num_classes)
        area of region enclosed by contour
    center : float-valued PyTorch tensor of size (batch_size, num_classes, 2)
        coordinates of center of contour
    """
    # Initialization
    center = torch.zeros(coeffs.shape[0:-1], dtype=torch.float).to(device)
    if len(deriv) == 0:
        deriv = diff(coeffs, period, device=device)
    if len(area) == 0:
        area = enclosed_area(coeffs, period, deriv=deriv, device=device)
    a1_conv_a2 = conv1d(coeffs[..., 0, :], coeffs[..., 1, :])

    # Compute contour integrals (up to sign)
    center[..., 0] = (
        torch.abs(period * zeroth_conv_term(a1_conv_a2, deriv[..., 0, :])) / area
    )
    center[..., 1] = (
        torch.abs(period * zeroth_conv_term(a1_conv_a2, deriv[..., 1, :])) / area
    )

    return center


def eval(coeffs: Tensor, period: float, pts: Tensor, device: str = "cpu") -> Tensor:
    """
    Evaluate Fourier series at prescribed gridpoints.

    Parameters
    ----------
    coeffs: complex-valued PyTorch tensor of size (num_components, order)
        fourier coefficients
    period: float
        period of function
    pts: float-valued PyTorch tensor of size (num_pts,)
        points at which the Fourier series will be evaluated

    Returns
    -------
    float-valued numpy array of size (num_components, num_pts)
        Fourier series evaluated at prescribd gridpoints
    """
    if len(pts) == 0:  # Evaluate on equispaced grid using FFT
        coeffs_sym = torch.cat(
            (torch.conj(coeffs), torch.flip(coeffs[..., 1:], dims=[-1])), dim=-1
        )
        return torch.real(fft.fft(coeffs_sym, dim=-1))
    elif len(pts) == 1 and pts[0] == 0:  # Evaluate at t=0
        return torch.real(coeffs[..., 0]) + 2 * torch.sum(
            torch.real(coeffs[..., 1:]), dim=-1
        )
    else:  # Evaluate at general non-equispaced grid
        num_dim = len(coeffs.size())
        order = coeffs.size()[-1]
        num_pts = pts.size()[0]
        freq = 2 * np.pi / period

        # Construct evaluation points
        grid = (
            freq
            * pts.view(num_pts, 1).repeat(1, order - 1).to(device)
            * torch.arange(1, order).view(1, order - 1).repeat(num_pts, 1).to(device)
        )

        # Mean
        a0 = torch.cat(
            num_pts * [coeffs[..., 0].unsqueeze(num_dim - 1)], dim=num_dim - 1
        )
        a0 = torch.real(a0)

        # ...
        mat_vec_str = "ij, kj -> ki"
        sum_cos = torch.einsum(
            mat_vec_str, torch.cos(grid), torch.real(coeffs[..., 1:])
        )

        sum_sin = torch.einsum(
            mat_vec_str, torch.sin(grid), torch.imag(coeffs[..., 1:])
        )

        return a0 + 2 * (sum_cos - sum_sin)
