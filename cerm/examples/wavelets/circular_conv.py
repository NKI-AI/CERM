"""Implementation n-d circular convolution."""

import torch
import torch.fft as fft
import numpy as np

from typing import List, Union
from torch import Tensor


def compute_order(seq_shape: Union[int, torch.Size]) -> Union[int, np.ndarray]:
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
    if isinstance(seq_shape, torch.Size):
        return np.array([(k + 1) // 2 for k in seq_shape])
    elif isinstance(seq_shape, int):
        return (seq_shape + 1) // 2


def pad_zeros(seq: Tensor, num_zeros: Union[List[int], Tensor]) -> Tensor:
    """
    Pad two-sided sequence with zeros.

    Parameters
    ----------
    seq: float-valued PyTorch Tensor of shape [batch_dims (2 * M - 1)]
        multivariate sequence of order M = (M_{1}, ..., M_{num_dim})
    num_zeros: Union[List[int], Tensor]
        number of zeros to append (two-sided) to last num_dim dimensions

    Returns
    -------
    float-valued PyTorch Tensor of shape [batch_dims (2 * K - 1)]
        multivariate sequence padded with zeros of order K = M + num_zeros
    """
    # Last num_dim dimensions are padded with zeros
    num_dim = len(num_zeros)

    # Check dimensions are consistent
    if len(seq.shape) < num_dim:
        raise ValueError(
            "Number dimensions input sequence and number of zeros do not match!"
        )

    pad = []
    for dim_idx in range(num_dim):
        pad = [num_zeros[dim_idx], num_zeros[dim_idx]] + pad

    return torch.nn.functional.pad(seq, pad)


def pad_periodic(a: Tensor, order_b: Tensor) -> Tensor:
    """
    Pad two-sided sequence a periodically for computation of a * b.
    Here we assume that a is a two-sided sequence of even size.

    Parameters
    ----------
    seq: float-valued PyTorch Tensor of shape [batch_dims 2 * M]
        multivariate sequence of even order M = (M_{1}, ..., M_{num_dim})
    order_b: int-valued PyTorch tensor of shape 2 * N - 1
        order of sequence with which a will be convolved

    Returns
    -------
    float-valued PyTorch Tensor of shape [batch_dims 2 * (M + N - 1)]
        multivariate sequence periodically padded
    """
    num_dim = len(order_b)
    num_batch_dim = len(a.shape) - num_dim
    order_a = compute_order(a.shape[num_batch_dim:])
    dim = tuple(range(num_batch_dim, num_batch_dim + num_dim))

    if torch.sum(torch.tensor(a.shape[num_batch_dim:]) % 2) > 0:
        raise NotImplementedError(
            "Periodic padding not yet supported for odd-sized arrays"
        )

    for curr_dim in dim:
        # Move current dimension to last dimension
        a = torch.transpose(a, curr_dim, -1)

        # Extract order sequence in current dimension
        order_a_curr_dim = order_a[curr_dim - num_batch_dim]
        order_b_curr_dim = order_b[curr_dim - num_batch_dim]

        # Periodic extension in current dimension
        a = torch.cat(
            (
                a[..., 2 * order_a_curr_dim + 1 - order_b_curr_dim :],
                a,
                a[..., 0 : order_b_curr_dim - 1],
            ),
            dim=-1,
        )

        # Move current dimension back
        a = torch.transpose(a, curr_dim, -1)

    return a


def truncate(seq: Tensor, num_dim: int, min_idx: Tensor, max_idx: Tensor) -> Tensor:
    """
    Truncate two-sided sequence using prescribed bounds on indices

    Parameters
    ----------
    seq: float-valued PyTorch tensor of shape [batch_dims (2 * M - 1)]
        two-sided multivariate sequence of order M = (M_{1}, ..., M_{num_dim})
    num_dim: int
        dimension of sequence
    min_idx: int-valued PyTorch tensor of dimension num_dim
        minimal index array (two-sided numbering)
    max_idx: int-valued PyTorch tensor of dimension num_dim
        maximal index array (two-sided numbering)

    Returns
    -------
    seq: float-valued PyTorch Tensor of shape [batch_dims max_idx - min_idx + 1]
        truncated sequence
    """
    num_batch_dim = len(seq.shape) - num_dim
    order = compute_order(seq.shape[num_batch_dim:])
    dim = tuple(range(num_batch_dim, num_batch_dim + num_dim))

    for curr_dim in dim:
        seq = torch.transpose(seq, curr_dim, -1)
        order_curr_dim = order[curr_dim - num_batch_dim]
        min_idx_curr_dim = min_idx[curr_dim - num_batch_dim]
        max_idx_curr_dim = max_idx[curr_dim - num_batch_dim]
        seq = seq[
            ...,
            order_curr_dim + min_idx_curr_dim - 1 : order_curr_dim + max_idx_curr_dim,
        ]
        seq = torch.transpose(seq, curr_dim, -1)

    return seq


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


def circular_conv(a: Tensor, b: Tensor, num_dim: int) -> Tensor:
    """
    Compute circular convolution of multivariate sequences a and b.

    Parameters
    ----------
    a: float-valued PyTorch tensor of size [batch_dims (2 * M - 1)]
        multivariate sequence of order M
    b: float-valued PyTorch tensor of size [batch_dims (2 * N - 1)]
        multivariate sequence or order N
    num_dim: int
        compute convolution over the last num_dim dimensions

    Returns
    -------
    a_conv_b: float-valued PyTorch tensor of size [batch_dims (2 * K - 1)]
        circular convolution a * b; sequence of order K = M + N - 1
    """
    if len(a.shape) != len(b.shape):
        raise ValueError("Dimensions of input sequences must match")

    if len(a.shape) < num_dim:
        raise ValueError(
            "Dimension input sequences and prescribed dimension do not match!"
        )

    # Determine two-sided orders of input sequences
    num_batch_dim = len(a.shape) - num_dim
    order_a = compute_order(a.shape[num_batch_dim:])
    order_b = compute_order(b.shape[num_batch_dim:])
    order_ab = order_a + order_b - 1

    # Pad sequences and convert to one-sided representation
    a_shift_pad = fftshift(pad_zeros(a, order_b - 1), num_dim)
    b_shift_pad = fftshift(pad_zeros(b, order_a - 1), num_dim)

    # Compute convolution by evaluating Laurent polynomial using DFT
    num_coeffs = (2 * order_ab - 1).tolist()
    dim = tuple(range(num_batch_dim, num_batch_dim + num_dim))
    dft_ab = fft.rfftn(a_shift_pad, dim=dim, s=num_coeffs) * fft.rfftn(
        b_shift_pad, dim=dim, s=num_coeffs
    )
    return fftshift(fft.irfftn(dft_ab, dim=dim, s=num_coeffs), num_dim, mode="inverse")


def circular_conv_periodic(a: Tensor, b: Tensor, num_dim: int) -> Tensor:
    """
    Compute periodic circular convolution of multivariate sequences a and b.

    Parameters
    ----------
    a: float-valued PyTorch tensor of size [batch_dims 2M]
        multivariate sequence of order M (even size)
    b: float-valued PyTorch tensor of size [batch_dims (2 * N - 1)]
        multivariate sequence or order N (odd size)
    num_dim: int
        compute convolution over the last num_dim dimensions

    Returns
    -------
    a_conv_b: float-valued PyTorch tensor of size [batch_dims 2M]
        (truncated) periodic circular convolution a * b
    """
    if len(a.shape) != len(b.shape):
        raise ValueError("Dimensions of input sequences must match")

    if len(a.shape) < num_dim:
        raise ValueError(
            "Dimension input sequences and prescribed dimension do not match!"
        )

    # Determine two-sided orders of input sequences
    num_batch_dim = len(a.shape) - num_dim
    order_a = compute_order(a.shape[num_batch_dim:])
    order_b = compute_order(b.shape[num_batch_dim:])
    dim = tuple(range(num_batch_dim, num_batch_dim + num_dim))

    # Periodic extension of a
    a_ext = pad_periodic(a, order_b)

    # Pad with one zero to the right to make order odd
    pad = []
    for dim_idx in range(num_dim):
        pad = [0, 1] + pad
    a_ext = torch.nn.functional.pad(a_ext, pad)

    # Compute convolution and truncate to original order of a
    a_conv_b = circular_conv(a_ext, b, num_dim)
    return truncate(a_conv_b, num_dim, -order_a, order_a - 1)
