"""Fourier tools."""

import torch
import numpy as np
import torch.fft as fft

from torch import Tensor


def reshape_to_odd_size(signal: Tensor) -> Tensor:
    """
    Reshape signal to odd size if necessary so that two-sided order is well-defined.

    Parameters
    ----------
    signal: float-valued PyTorch tensor of size [batch_size num_channels heigth width]
        signal at some initial resolution level
    """
    spatial_dim = signal.shape[-2:]
    pad = []
    for dim in spatial_dim:
        if dim % 2 == 0:
            pad = [0, 1] + pad
        else:
            pad = [0, 0] + pad
    return torch.nn.functional.pad(signal, pad)


def compute_order(seq_shape: torch.Size) -> Tensor:
    """
    Determine order of two-sided multivariate sequence.

    Parameters
    ----------
    seq_shape: torch.Size
        shape of two-sided sequence

    Returns
    -------
    int-valued Pytorch Tensor
        order M of two-sided sequence
    """
    return torch.div(torch.tensor(seq_shape) + 1, 2, rounding_mode="trunc")


def fftshift(
    seq: Tensor, num_dim: int, mode: str = "forward", in_place: bool = False
) -> Tensor:
    """
    Map two-sided even sequence to equivalent one-sided form to apply DFT.

    Parameters
    ----------
    seq: float-valued PyTorch Tensor of shape [batch_dims (2 * M - 1)]
        two-sided multivariate sequence of order M = (M_{1}, ..., M_{num_dim})
    num_dim: int
        perform fftshift over last num_dim dimensions
    mode: str
        perform forward or inverse fft shift
    in_place: bool
        if true the fftshift is performed in place

    Returns
    -------
    float-valued Pytorch tensor of size [batch_dims (2 * M - 1)]
        one-sided representation of sequence
    """
    # Determine number batch dimensions and order of sequences
    num_batch_dim = len(seq.shape) - num_dim
    order = compute_order(seq.shape[num_batch_dim:])
    dim = tuple(range(num_batch_dim, num_batch_dim + num_dim))

    if not in_place:
        seq = seq.clone()

    if mode not in ["forward", "inverse"]:
        raise NotImplementedError("Choose either forward or inverse mode")

    for curr_dim in dim:
        # Swap current dimension and last one
        seq = torch.transpose(seq, curr_dim, -1)
        order_curr_dim = order[curr_dim - num_batch_dim]

        if mode == "forward":
            idx = list(range(order_curr_dim - 1, 2 * order_curr_dim - 1)) + list(
                range(0, order_curr_dim - 1)
            )

        elif mode == "inverse":
            idx = list(range(order_curr_dim, 2 * order_curr_dim - 1)) + list(
                range(0, order_curr_dim)
            )

        # Place coefficients with negative indices after positive ones
        seq[..., 0 : 2 * order_curr_dim - 1] = seq[..., idx]
        seq = torch.transpose(seq, curr_dim, -1)

    return seq


def fourier_coeffs(feval: Tensor, tol: float = 1e-04) -> Tensor:
    """
    Compute Fourier coefficients of function evaluated at equispaced grid

    Parameters
    ----------
    feval: float-valued PyTorch Tensor of shape [batch_dims 2N1-1 2N2-1]
        (L1, L2)-periodic map evaluated on equispaced grid

    Returns
    -------
    """
    feval_odd = reshape_to_odd_size(feval)
    spatial_dim = torch.tensor(feval_odd.shape[-2:])
    coeffs = fft.fft2(reshape_to_odd_size(feval)) / torch.prod(spatial_dim)
    coeffs[torch.abs(coeffs) < tol] = 0.0
    return fftshift(coeffs, len(spatial_dim), mode="inverse")


class FourierSeries2d:
    """Class representing Fourier series."""

    def __init__(self, coeffs: Tensor, period: float) -> None:
        """
        Initialize Fourier series (not assumed to be real-valued)

        Parameters
        ----------
        coeffs: complex-valued PyTorch Tensor of size [num_series 2N1-1 2N2-1]
            Fourier series
        period: float-valued PyTorch Tensor of size [2]
            period of Fourier series
        """
        self.coeffs = coeffs.type(torch.complex64)
        self.order = compute_order(coeffs.shape[-2:])
        self.period = period
        self.angular_freq = 2 * np.pi / self.period
        self.num_spatial_components = 2

    def diff(self) -> None:
        """Differentiate Fourier series"""
        device = self.coeffs.device

        if self.coeffs.shape[0] > 1:
            raise ValueError(
                "Differentiation is only supported for scalar-valued Fourier series"
            )

        coeffs = self.coeffs.squeeze(0)

        # Multiplication factor for differentiation in frequency domain
        factor = [
            1j * self.angular_freq[idx] * torch.arange(1 - N, N).to(device)
            for idx, N in enumerate(self.order)
        ]

        #
        coeffs_diff = torch.stack(
            [
                torch.einsum("i, ij -> ij", factor[0], coeffs),
                torch.einsum("j, ij -> ij", factor[1], coeffs),
            ]
        )

        return FourierSeries2d(coeffs_diff, self.period.clone())

    def __call__(self, points: Tensor) -> Tensor:
        """
        Evaluate Fourier series at collection of arbitrary points.

        Parameters
        ----------
        points: float-valued PyTorch Tensor of shape [num_pts 2]
            points to evaluate image at

        Returns
        -------
        vals: float-valued PyTorch Tensor of shape [num_series num_pts]
            batch of series evaluated at prescribed points
        """
        device = points.device
        modes = [torch.arange(1 - N, N).float().to(device) for N in self.order]

        # Evaluate exponentials at each spatial component
        exp_eval = [
            torch.exp(
                1j
                * self.angular_freq[spatial_idx]
                * (
                    points[:, spatial_idx].unsqueeze(1)
                    @ modes[spatial_idx].unsqueeze(0)
                )
            )
            for spatial_idx in range(self.num_spatial_components)
        ]

        # Evaluate sums in series iteratively to avoid construction large arrays

        # Evaluate Fourier series last dimension ("inner sum")
        vals = torch.einsum("jmn, ln -> jlm", self.coeffs, exp_eval[1])

        # Evaluate Fourier series first dimension ("outer sum")
        return torch.real(torch.einsum("jlm, lm -> jl", vals, exp_eval[0]))
