"""Discrete wavelet transform (1d and separable 2d) and tools"""

import numpy as np
import torch
import warnings

from typing import Tuple, List, Dict, Union
from torch import Tensor

from cerm.examples.wavelets.circular_conv import (
    compute_order,
    circular_conv,
    circular_conv_periodic,
)


def exponent_base_two(n: int) -> int:
    """
    Check whether n = 2^k for some k.

    Parameters
    ----------
    n: int
        integer to check whether is power of two

    Returns
    -------
    int
        exponent of integer if it is of the form 2^k
    """
    if (n & n - 1 == 0) and n != 0:
        return int(round(np.log(n) / np.log(2)))
    else:
        raise ValueError("Number of coefficients is not a power of two")


def determine_order_parity(init_dim: Tuple[int, int]) -> Tuple[bool, bool]:
    """
    Determine parity of order associated to dimensions

    Parameters
    ----------
    init_dim: Tuple[int, int]
        (spatial) dimensions initial signal
    """
    num_spatial_dim = len(init_dim)
    dim = [*init_dim]
    for dim_idx in range(num_spatial_dim):
        if dim[dim_idx] % 2 == 0:
            dim[dim_idx] += 1
    return compute_order(dim) % 2


def compute_dims_dwt(
    init_dim: Tuple[int, int],
    order_filter: Union[int, Tuple[int, int]],
    num_levels: int,
) -> List[List[int]]:
    """
    Compute dimensions of coefficients in wavelet decomposition

    Parameters
    ----------
    init_dim: Tuple[int, int]
        (spatial) dimensions initial signal
    order_filter: Tuple[int, int]
        order of filters in each coordinate direction
    num_levels: int
        number of decomposition levels

    Returns
    -------
    List[Tensor]
        list containing spatial dimensions per resolution level
    """
    # Dimensions
    num_spatial_dim = len(init_dim)

    # We increase order by one in dwt so that hpf and lpf have same order
    order_filter = torch.tensor(order_filter).int() + 1

    # In wavelet decomposition we ensure length initial signal is odd
    dim = [*init_dim]
    for dim_idx in range(num_spatial_dim):
        if dim[dim_idx] % 2 == 0:
            dim[dim_idx] += 1
    dim_per_level = [torch.tensor(dim).int()]

    for level in range(num_levels):

        down_conv_order = []

        # Order after convolution with filter
        order_coeffs = compute_order(dim_per_level[level])
        order_conv = order_coeffs + order_filter - 1

        # Order after downsampling; depends on parity of order
        for dim_idx in range(num_spatial_dim):
            if order_conv[dim_idx] % 2 == 0:
                down_conv_order.append(
                    torch.div(order_conv[dim_idx], 2, rounding_mode="floor")
                )

            else:
                down_conv_order.append(
                    torch.div(order_conv[dim_idx] + 1, 2, rounding_mode="floor")
                )

        dim_per_level.append(torch.tensor([2 * order - 1 for order in down_conv_order]))

    return [dim.tolist() for dim in dim_per_level]


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


def repeat_filter(filter: Tensor, batch_size: int, spatial_dim_size: int) -> Tensor:
    """
    Repeat filter by adding a batch dimension and spatial dimension

    Parameters
    ----------
    filter: float-valued PyTorch tensor of size [num_channels num_filters 2m-1]
        two-sided representation of filter
    batch_size: int
        number of signals
    spatial_dim_size: int
        size of spatial dimension

    Returns
    -------
    filter_rep: float-valued Pytorch tensor [batch_size num_channels num_filters num_spatial_dim 2m-1]
        repeated filter
    """
    num_channels = filter.shape[0]
    num_filters = filter.shape[1]
    len_filter = filter.shape[-1]
    return filter.view(1, num_channels, num_filters, 1, len_filter).repeat(
        batch_size, 1, 1, spatial_dim_size, 1
    )


def high_pass_filter(lpf: Tensor) -> Tensor:
    """
    Construct Mallat's high pass filter for a given low pass filter.

    Parameters
    ----------
    lpf: float-valued PyTorch tensor of size [num_filters 2m-1]
        two-sided representation of low pass filter

    Returns
    -------
    hpf: float-valued PyTorch tensor of size [num_filters 2m+1]
        high pass filter
    """
    num_channels = lpf.shape[0]
    num_filters = lpf.shape[1]
    m = int(compute_order(lpf.shape[-1]))
    k = torch.arange(2 - m, m + 1).to(lpf.device)
    hpf = torch.zeros(num_channels, num_filters, 2 * m + 1).to(lpf.device)
    hpf[:, :, 2:] = torch.einsum(
        "ijk, k -> ijk", torch.flip(lpf, dims=[-1]), (-1) ** (k - 1)
    )
    return hpf


def downsample_1d(coeffs: Tensor) -> Tensor:
    """
    Extract coefficients with even index from last dimension.

    Parameters
    ----------
    a: float-valued Pytorch tensor of size [ ... 2n-1]
        approximation coefficients at some resolution level

    Returns
    -------
    float-valued Pytorch tensor of size
        coefficients with only even indices
    """
    order = int(compute_order(coeffs.shape[-1]))
    if order % 2 == 0:
        # Terms even index: [a_{2 - K} ... a_{K-2}] steps of two; order K / 2
        return coeffs[..., slice(1, 2 * order - 2, 2)]
    else:
        # Terms even index: [a_{1 - K} ... a_{K-1}] steps of two; order (K+1) / 2
        return coeffs[..., slice(0, 2 * order - 1, 2)]


def downsample_1d_periodic(a: Tensor) -> Tensor:
    """
    Extract coefficients with even index from even-sized sequence

    Parameters
    ----------
    a: float-valued Pytorch tensor of size [ ... 2 * n]
        two-sided sequence of even order

    Returns
    -------
    float-valued Pytorch tensor of size
        coefficients with only even indices
    """
    return a[..., slice(0, a.shape[-1], 2)]


def upsample_1d(coeffs: Tensor) -> Tuple[int, Tensor]:
    """
    Upsample array by placing elements of sequence at even indices.

    Parameters
    ----------
    coeffs: float-valued Pytorch tensor of size [... 2n-1]
        approximation or detail coefficients at some resolution level

    Returns
    -------
    order_up: int
        order after upsampling
    coeffs_up: float-valued Pytorch tensor of size [... 2(2n-1)-1]
        coefficients are upsampling
    """
    order = int(compute_order(coeffs.shape[-1]))
    order_up = 2 * order - 1
    coeffs_up = torch.zeros(*coeffs.shape[0:-1], 2 * order_up - 1).to(coeffs.device)
    coeffs_up[..., slice(0, 2 * order_up - 1, 2)] = coeffs
    return order_up, coeffs_up


def upsample_1d_periodic(a: Tensor) -> Tensor:
    """
    Upsample periodic array by placing elements of sequence at even indices.

    Parameters
    ----------
    a: float-valued Pytorch tensor of size [ ... n]
        approximation coefficients at some resolution level

    Returns
    -------
    float-valued Pytorch tensor of size
    """
    n = a.shape[-1]
    y = torch.zeros(*a.shape[0:-1], 2 * n).to(a.device)
    y[..., slice(0, 2 * n + 1, 2)] = a
    return y


def down_conv1d(coeffs: Tensor, filter: Tensor, num_dim_conv: int = 1) -> Tensor:
    """
    Convolve coefficients with low or high pass filter in last dimension and down sample.

    Parameters
    ----------
    coeffs float-valued Pytorch Tensor of size [... 2n-1]
        approximation or detail coefficients at some resolution level
    filter: float-valued PyTorch Tensor of size [... 2m-1]
        low or high pass filter
    num_dim_conv: int
        number of spatial dimensions convolution

    Returns
    -------
    float-valued Pytorch Tensor of size [...., 2s-1]
        down-sampled signal after 1d convolution; order is s=K/2 or s=(K+1)/2, K=m+n-1
    """
    return downsample_1d(circular_conv(coeffs, filter, num_dim_conv))


def down_conv1d_periodic(
    coeffs: Tensor, filter: Tensor, num_dim_conv: int = 1
) -> Tensor:
    """
    Convolve coefficients with low or high pass filter in last dimension and down sample.

    Parameters
    ----------
    coeffs float-valued Pytorch Tensor of size [... 2n]
        approximation or detail coefficients at some resolution level
    filter: float-valued PyTorch Tensor of size [... 2m-1]
        low or high pass filter
    num_dim_conv: int
        number of spatial dimensions convolution

    Returns
    -------
    float-valued Pytorch Tensor of size [...., n]
        down-sampled signal after 1d convolution
    """
    return downsample_1d_periodic(circular_conv_periodic(coeffs, filter, num_dim_conv))


def conv_up_1d(
    coeffs: Tensor, filter: Tensor, order_up_parity: bool, num_dim_conv: int = 1
) -> Tensor:
    """
    Perform convolution with high or low pass filter and upsample.

    Parameters
    ----------
    coeffs: float-valued Pytorch Tensor of size [... 2s-1]
        approximation or detail coefficients at some resolution level
    filter: float-valued PyTorch Tensor of size [... 2m-1]
        low or high pass filter
    order_up_parity: bool
        indicates whether order of conv(upsample(coeffs), filter) is odd or even

    Returns
    -------
    float-valued Pytorch Tensor of size [...., 2s-1]
        filter convoled with upsampled coefficients
    """
    order_filter = compute_order(filter.shape[-1])
    order_coeffs_up, coeffs_up = upsample_1d(coeffs)

    # Recover order coefficients one level up
    if ((order_up_parity == 0) and (order_filter % 2 == 1)) or (
        (order_up_parity == 1) and (order_filter % 2 == 0)
    ):
        order_original_coeffs = order_coeffs_up - order_filter + 2
    else:
        order_original_coeffs = order_coeffs_up - order_filter + 1

    # Find position in conv(upsample(a), filter) corresponding to index 1 - N
    idx_0 = order_coeffs_up + order_filter - 1 - order_original_coeffs

    # Convolve and keep non-zero coefficients
    return circular_conv(coeffs_up, filter, num_dim_conv)[
        ..., idx_0 : idx_0 + 2 * order_original_coeffs - 1
    ]


def conv_up_1d_periodic(
    coeffs: Tensor, filter: Tensor, order_up_parity: bool, num_dim_conv: int = 1
) -> Tensor:
    """
    Perform convolution with high or low pass filter and upsample.

    Parameters
    ----------
    coeffs: float-valued Pytorch Tensor of size [... s]
        approximation or detail coefficients at some resolution level
    filter: float-valued PyTorch Tensor of size [... 2m-1]
        low or high pass filter
    order_up_parity: bool
        indicates whether order of conv(upsample(coeffs), filter) is odd or even

    Returns
    -------
    float-valued Pytorch Tensor of size [...., 2s]
        filter convoled with upsampled coefficients
    """
    return circular_conv_periodic(upsample_1d_periodic(coeffs), filter, num_dim_conv)


def down_conv2d_separable(
    coeffs: Tensor,
    filter: Union[Tuple[Tensor, Tensor], List[Tensor]],
    periodic_signal: bool = False,
) -> Tensor:
    """
    Perform separable convolution in both spatial dimensions and down sample.

    Parameters
    ----------
    coeffs: float-valued Pytorch Tensor of size [...., 2n1 - 1, 2n2 - 1]
        approximation or detail coefficients at some resolution level
    filter: Tuple[Tensor, Tensor]
        tuple of low-high pass filters
    periodic_signal: bool
        indicates whether the input signal is periodic

    Returns
    -------
    float-valued Pytorch Tensor of size [...., 2s-1, 2s-1]
        down-sampled signal after 2d (separable) convolution
    """
    # Initialize dimensions and appropriate convolution operator
    batch_size = coeffs.shape[0]
    spatial_dim_size = coeffs.shape[-2:]
    if periodic_signal:
        dwt1d = down_conv1d_periodic
    else:
        dwt1d = down_conv1d

    # Convolution in last dimension
    filter_rep = repeat_filter(filter[1], batch_size, spatial_dim_size[0])
    y = dwt1d(coeffs, filter_rep)

    # Convolution in penultimate dimension
    filter_rep = repeat_filter(filter[0], batch_size, y.shape[-1])
    y = dwt1d(torch.swapaxes(y, -1, -2), filter_rep)

    return torch.swapaxes(y, -1, -2)


def conv2d_up_separable(
    coeffs: Tensor,
    filter: Union[Tuple[Tensor, Tensor], List[Tensor]],
    order_up_parity: Tuple[bool, bool],
    periodic_signal: bool = False,
) -> Tensor:
    """
    Up sample signal in both spatial dimensions and perform separable convolution

    Parameters
    ----------
    coeffs: float-valued Pytorch Tensor of size [...., 2n1 - 1, 2n2 - 1]
        signal (or approximation coefficients) at some resolution level
    filter: Tuple[Tensor, Tensor]
        tuple of low-high pass filters
    order_up_parity: Tuple[bool, bool]
        indicates whether order of conv(upsample(coeffs), filter) is odd or even
    periodic_signal: bool
        indicates whether the input signal is periodic

    Returns
    -------
    float-valued Pytorch Tensor of size [...., 2s1-1, 2s2-1]
        up-sampled signal after 2d (separable) convolution
    """
    # Initialize dimensions and appropriate convolution operator
    batch_size = coeffs.shape[0]
    spatial_dim_size = coeffs.shape[-2:]
    if periodic_signal:
        idwt1d = conv_up_1d_periodic
    else:
        idwt1d = conv_up_1d

    # Convolution in last dimension
    filter_rep = repeat_filter(filter[1], batch_size, spatial_dim_size[0])
    y = idwt1d(coeffs, filter_rep, order_up_parity[1])

    # Convolution in penultimate dimension
    filter_rep = repeat_filter(filter[0], batch_size, y.shape[-1])
    y = idwt1d(torch.swapaxes(y, -1, -2), filter_rep, order_up_parity[0])

    return torch.swapaxes(y, -1, -2)


def idwt1d(
    init_approx: Tensor,
    detail: List[Tensor],
    lpf: Tensor,
    parity_init_order: int = 1,
    num_levels_no_detail: int = 0,
    periodic_signal: bool = False,
) -> List[Tensor]:
    """
    Inverse DWT 1d

    Parameters
    ----------
    init_approx: List[Tensor]
        approximation coefficients at initial resolution level
    detail: List[Tensor]
        detail coefficients at initial and higher resolution levels
    num_level_no_detail: int
        number of resolution levels up after exhausting provided detail coefficients
    periodic_signal: bool
        indicates whether the input signal is periodic

    Returns
    -------
    approx: List[Tensor]
        reconstructed approximation coefficients at higher resolution levels
    """
    batch_size = init_approx.shape[0]

    hpf = high_pass_filter(lpf)
    lpf = torch.nn.functional.pad(lpf, (1, 1))  # match order with hpf

    # Repeat filters
    hpf = hpf.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    lpf = lpf.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    num_levels = len(detail)
    approx = [init_approx]

    if periodic_signal:
        upsample = conv_up_1d_periodic
    else:
        upsample = conv_up_1d

    # idwt
    for level in range(num_levels):

        if level <= num_levels - 2:
            order_up_parity = compute_order(detail[level + 1].shape[-1]) % 2
        else:
            order_up_parity = parity_init_order

        approx.append(
            upsample(approx[-1], lpf, order_up_parity)
            + upsample(detail[level], hpf, order_up_parity)
        )

    # Reconstruction without detail coefficients
    for level in range(num_levels_no_detail):
        approx.append(upsample(approx[-1], lpf, parity_init_order))

    return approx


def dwt1d(
    signal: Tensor,
    lpf: Tensor,
    num_levels: int,
    num_spatial_dim: int = 1,
    periodic_signal: bool = False,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Discrete Wavelet Transform (1d)

    Parameters
    ----------
    signal: float-valued PyTorch tensor of size [batch_size num_channels signal_size]
        approximation coefficients at initial resolution level
    lpf: Tensor
        low-high pass filters of shape [num_channels num_filters 2m-1]
    num_levels: int
        number of resolution levels
    periodic_signal: bool
        indicates whether the input signal is periodic

    Returns
    -------
    approx: List[Tensor]
        approximation coefficients at lower resolution levels
    List[Tensor]
        detail coefficients
    """
    # Dimensions
    batch_size = signal.shape[0]
    num_channels = signal.shape[1]
    num_filters = lpf.shape[1]
    if periodic_signal and signal.shape[-1] % 2 == 1:
        raise NotImplementedError(
            "Only even-shaped signals are supported in periodic case"
        )

    if not periodic_signal and signal.shape[-1] % 2 == 0:
        raise NotImplementedError("Only odd-shaped non-periodic signals are supported")

    # Flippped low and high pass filters
    hpf_flip = torch.flip(high_pass_filter(lpf), dims=[-1])
    lpf_flip = torch.flip(lpf, dims=[-1])
    lpf_flip = torch.nn.functional.pad(lpf_flip, (1, 1))  # match order of hpf

    # Repeat filters (batch-size)
    lpf_flip = lpf_flip.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    hpf_flip = hpf_flip.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Repeat signal for each filter
    signal = torch.permute(
        signal.unsqueeze(0).repeat(num_filters, 1, 1, 1), (1, 2, 0, 3)
    )

    # Preallocation
    approx = [signal]
    detail = []

    # Determine which convolution (padding) to be used
    if periodic_signal:
        orth_proj = down_conv1d_periodic
    else:
        orth_proj = down_conv1d

    # DWT
    for level in range(num_levels):

        # Detail coefficients
        detail.insert(
            0,
            orth_proj(
                approx[0],
                hpf_flip,
            ),
        )

        # Approximation coefficients
        approx.insert(
            0,
            orth_proj(approx[0], lpf_flip),
        )

    return approx, detail


def dwt2d(
    signal: Tensor,
    lpf: Tuple[Tensor, Tensor],
    num_levels: int,
    num_spatial_dim: int = 2,
    periodic_signal: bool = False,
) -> Tuple[List[Tensor], Dict[str, List[Tensor]]]:
    """
    Discrete Wavelet Transform (2d)

    Parameters
    ----------
    signal: float-valued PyTorch tensor of size [batch_size num_channels heigth width]
        approximation coefficients at initial resolution level
    lpf: Tuple[Tensor, Tensor]
        tuple of low-high pass filters each of shape [num_channels num_filters 2m-1]
    num_levels: int
        number of resolution levels
    periodic_signal: bool
        indicates whether the input signal is periodic

    Returns
    -------
    approx: List[Tensor]
        approximation coefficients at lower resolution levels
    Dict[str, List[Tensor]]
        detail coefficients with respect to decomposition W11, W12 and W21
    """
    # Dimensions
    num_filters = lpf[0].shape[1]
    if periodic_signal and (signal.shape[-1] % 2 == 1 or signal.shape[-2] % 2 == 1):
        raise NotImplementedError(
            "Only even-shaped signals are supported in periodic case"
        )

    if not periodic_signal and (signal.shape[-1] % 2 == 0 or signal.shape[-2] % 2 == 0):
        warnings.warn("Signal is even-shaped and not periodic; reshaping to odd-size")
        signal = reshape_to_odd_size(signal)

    # Flippped low and high pass filters
    hpf_flip = [torch.flip(high_pass_filter(y), dims=[-1]) for y in lpf]
    lpf_flip = [torch.flip(y, dims=[-1]) for y in lpf]
    lpf_flip = [
        torch.nn.functional.pad(y, (1, 1)) for y in lpf_flip
    ]  # match order of hpf

    # Repeat signal for each filter
    signal = signal.unsqueeze(2).repeat(1, 1, num_filters, 1, 1)

    # Preallocation
    approx = [signal]
    d12 = []
    d21 = []
    d22 = []

    # DWT
    for level in range(num_levels):

        # Detail coefficients: W12
        d12.insert(
            0,
            down_conv2d_separable(
                approx[0], [lpf_flip[0], hpf_flip[1]], periodic_signal=periodic_signal
            ),
        )

        # Detail coefficients: W21
        d21.insert(
            0,
            down_conv2d_separable(
                approx[0], [hpf_flip[0], lpf_flip[1]], periodic_signal=periodic_signal
            ),
        )

        # Detail coefficients: W22
        d22.insert(
            0,
            down_conv2d_separable(approx[0], hpf_flip, periodic_signal=periodic_signal),
        )

        # Approximation coefficients
        approx.insert(
            0,
            down_conv2d_separable(approx[0], lpf_flip, periodic_signal=periodic_signal),
        )

    return approx, {"d12": d12, "d21": d21, "d22": d22}


def idwt2d(
    init_approx: Tensor,
    detail: Dict[str, Tensor],
    lpf: Tuple[Tensor, Tensor],
    parity_init_order: Tuple[int, int] = (1, 1),
    num_levels_no_detail: int = 0,
    periodic_signal: bool = False,
) -> List[Tensor]:
    """
    Inverse DWT (2d).

    Parameters
    ----------
    approx: float-valued PyTorch Tensor of size [... 2n1-1 2n2-1]
        approximation coefficients at a lowest resolution level
    detail: Dict[str, Tensor]
        detail coefficients
    lpf: Tuple[Tensor, Tensor]
        tuple of low-high pass filters each of shape [num_filters 2m-1]
    order_up_parity: Tuple[bool, bool]
        indicates whether order of original signal should be odd or even
    num_level_no_detail: int
        number of resolution levels up after exhausting provided detail coefficients
    periodic_signal: bool
        indicates whether the input signal is periodic

    Returns
    -------
    """
    if not (len(detail["d12"]) == len(detail["d21"]) == len(detail["d22"])):
        raise ValueError("Different number of levels contained in details")
    num_levels = len(detail["d12"])

    # Initialize low and high pass filters
    hpf = [high_pass_filter(y) for y in lpf]
    lpf = [torch.nn.functional.pad(y, (1, 1)) for y in lpf]  # match order with hpf

    # IDWT
    approx = [init_approx]
    for level in range(num_levels):

        if level <= num_levels - 2:
            order_up_parity = compute_order(detail["d12"][level + 1].shape[-2:]) % 2
        else:
            order_up_parity = parity_init_order

        approx.append(
            conv2d_up_separable(
                approx[-1], lpf, order_up_parity, periodic_signal=periodic_signal
            )
            + conv2d_up_separable(
                detail["d12"][level],
                (lpf[0], hpf[1]),
                order_up_parity,
                periodic_signal=periodic_signal,
            )
            + conv2d_up_separable(
                detail["d21"][level],
                (hpf[0], lpf[1]),
                order_up_parity,
                periodic_signal=periodic_signal,
            )
            + conv2d_up_separable(
                detail["d22"][level],
                hpf,
                order_up_parity,
                periodic_signal=periodic_signal,
            )
        )

    # Reconstruction without detail coefficients
    for level in range(num_levels_no_detail):
        approx.append(
            conv2d_up_separable(
                approx[-1], lpf, parity_init_order, periodic_signal=periodic_signal
            )
        )

    return approx
