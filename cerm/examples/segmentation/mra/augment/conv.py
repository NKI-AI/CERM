"""Module with tools for computing convolution and related quantities."""

import numpy as np
import torch
from scipy.stats import norm


def mult2ind(mult_ind, dim_size):
    """
    Convert d-dimensional multi-index to linear index given array size using row-major ordering.

    Parameters
    ----------
    mult_ind: int-valued Pytorch tensor or list of size d x N
        the N multi-indices (stored amongst columns) to be converted to linear indices
    dim_size: list of ints of length d
        size of each dimension associated to array

    Returns
    -------
    int-valued Pytorch tensor of size (N,)
        linear indices corresponding to multi-indices
    """
    return mult_ind[0, :] * dim_size[1] + mult_ind[1, :]


def complex_mult(u, v):
    """
    Multiply two complex-valued arrays.

    Parameters
    ----------
    u: float-valued Pytorch tensor of size [ .... 2]
        array of arbitrary size, where the last dimension corresponds to the real and imaginary
        components
    v: float-valued Pytorch tensor of size [ .... 2]
        array of same size as u, where the last dimension corresponds to the real and imaginary
        components

    Returns
    -------
    float-valued pytorch tensor of same size as input
        component-wise (complex) product of x and y
    """
    # Switch first and last dimension
    perm = list(range(len(u.size())))
    perm[0] = perm[-1]
    perm[-1] = 0
    u = u.permute(perm)
    v = v.permute(perm)

    # Multiply
    a_dot_b = torch.zeros(*u.size())
    a_dot_b[0] = u[0] * v[0] - u[1] * v[1]
    a_dot_b[1] = u[0] * v[1] + u[1] * v[0]

    return a_dot_b.permute(perm)


def two_sided_order(a):
    """
    Determine order n associated to a sequence.

    .. math::
        a = [a_{1-n}  ...  a_{0}  ...  a_{n}]

    if n is even and

    .. math::
        a = [a_{1-n}  ...  a_{0}  ...  a_{n-1]}]

    if n is odd.

    Parameters
    ----------
    a: float-valued Pytorch tensor of size [batch_size seq_length]

    Returns
    -------
    int
        two-sided order associated to sequence
    """
    seq_len = a.size()[1]
    if seq_len % 2 == 0:
        return seq_len // 2
    else:
        return (seq_len + 1) // 2


def dft_shift(coeffs, mode="forward"):
    """
    Map two-sided sequence to "equivalent" one-sided form for using DFT.

    Parameters
    ----------
    coeffs: float-valued Pytorch tensor of size [batch_size seq_len]
        each row is a sequence of the form [a_{0} ... a_{seq_len - 1}]
    mode: str in {forward, inv}
        indicates whether to perform the forward or inverse mapping

    Returns
    -------
    float-valued Pytorch tensor of size [batch_size sequence_length]
        each row is a sequence of the form []

    """
    n = coeffs.size()[1]
    mid_idx = (n - 1) // 2
    if mode == "forward" or (mode == "inverse" and n % 2 == 0):
        return torch.cat((coeffs[:, mid_idx + 1 :], coeffs[:, : mid_idx + 1]), 1)
    elif mode == "inverse":
        return torch.cat((coeffs[:, mid_idx:], coeffs[:, :mid_idx]), 1)
    else:
        raise NotImplementedError("Choose either forward or inverse mapping")


def pad(coeffs, num_coeffs):
    """
    Pad symmetric sequence with zeros to increase length to prescribed size.

    Parameters
    ----------
    coeffs: float-valued Pytorch tensor of size [batch_size sequence_length]
        sequence of coefficients [a_{1-n} ... a_{0} ... a_{n-1}]
    num_coeffs: int
        length to which sequence should be increased

    Returns
    -------
    coeffs_pad: float-valued Pytorch tensor of size [batch_size num_coeffs]
        sequence of coefficients [0 ... 0 a_{1-n} ... a_{0} ... a_{n-1} 0 ... 0]
    """
    dim = coeffs.size()
    num_zeros = num_coeffs - dim[1]
    z = torch.zeros((dim[0], num_zeros // 2))
    if num_zeros % 2 == 0:
        return torch.cat((z, coeffs, z), 1)
    else:
        return torch.cat((z, coeffs, z, torch.zeros(dim[0], 1)), 1)


def toeplitz_matrix(a, num_shifts):
    """
    Construct Toeplitz matrix (currently only implemented for sequences of odd length).

    Parameters
    ----------
    a: float-valued Pytorch tensor of size [1 num_coeffs]
        sequence of coefficients interpreted as [a_{1-n} ... a_{0} ... a_{n-1}] if n is odd and
        [a_{1-n} ... a_{0} ... a_{n}] otherwise
    num_shifts: int
        number of times to shift the sequence a to the right

    Returns
    -------
    toeplitz: float-valued Pytorch tensor of size [num_coeffs num_coeffs]
        Truncated Toeplitz matrix representing the map w -> Pi_{num_coeffs}(a * w)
    """
    # Dimensions
    order_a = two_sided_order(a)
    len_a = a.shape[1]

    # Construct linear indices associated to positions shifted row
    rows = (
        torch.arange(num_shifts)
        .view(num_shifts, 1)
        .repeat(1, len_a)
        .view(num_shifts * len_a)
    )
    cols = torch.arange(len_a).view(1, len_a).repeat(num_shifts, 1) + torch.arange(
        num_shifts
    ).view(num_shifts, 1)
    cols = cols.view(num_shifts * len_a)
    mult_ind = torch.stack((rows, cols), 0)
    lin_ind = mult2ind(mult_ind, [num_shifts, num_shifts + len_a - 1])

    # Construct toeplitz matrix
    toeplitz = torch.zeros(num_shifts * (num_shifts + len_a - 1))
    toeplitz[lin_ind] = a.flatten()[torch.arange(len_a - 1, -1, -1)].repeat(num_shifts)
    toeplitz = toeplitz.view([num_shifts, num_shifts + len_a - 1])

    # Extract relevant columnns
    return toeplitz[:, order_a - 1 : order_a + num_shifts - 1]


def conv1d(a, b, mode="trunc", method="toeplitz"):
    """
    Compute 1d two-sided convolution a * b.

    Coefficients are interpreted as

    .. math::
        [a_{(., 1-n)}  ...  a_{(., 0)}  ...  a_{(., n-1)}]

    if n is odd and

    .. math::
        [a_{(., 1-n)}  ...  a_{(., 0)}  ...  a_{(.,n)}]

    if n is even (the same holds for b). If the chosen method is FFT, then a_{i,:} * b_{i, :} is
    computed for 0 <= i <= batch_size - 1. If the chosen method if Toeplitz, then b should only
    have one row; in this case a_{i, :} * b is computed for 0 <= i <= batch_size - 1. Currently,
    only the truncated convolution is implemented for the Toeplitz-approach.

    Parameters
    ----------
    a: float-valued Pytorch tensor of size [batch_size num_coeffs]
        sequence of coefficients
    b: float-valued Pytorch tensor of size [batch_size num_coeffs]
        sequence of coefficients
    mode: str in {full, trunc}, optional
        compute all convolution terms or truncate to nonzero-coefficients determined by (maximal)
        length of input
    method: str in {toeplitz, fft, torch}, optional
        method for computing convolution

    Returns
    -------
    a_conv_b: float-valued Pytorch tensor of size (batch_size, ...)
        nonzero coefficients of two-sided convolution a * b
    """
    # Initialize dimensions
    dim_a = a.size()
    len_a = a.size()[1]
    len_b = b.size()[1]
    order_a = two_sided_order(a)
    order_b = two_sided_order(b)
    order_ab = 2 ** int(
        torch.ceil(
            torch.log(torch.tensor(2.0 * (order_a + order_b) - 1))
            / torch.log(torch.tensor(2.0))
        )
    )

    if method == "fft":
        # Pad with zeros, convert to one-sided representation and compute convolution
        a_hat = torch.rfft(dft_shift(pad(a, order_ab), "forward"), 1, onesided=False)
        b_hat = torch.rfft(dft_shift(pad(b, order_ab), "forward"), 1, onesided=False)
        a_conv_b = dft_shift(
            torch.irfft(complex_mult(a_hat, b_hat), 1, onesided=False), "inverse"
        )

        # Determine nonzero coefficients
        if mode == "full":
            min_index = order_ab // 2 - order_a - order_b
            max_index = order_ab // 2 + order_a + order_b - 2
            if (len_a % 2 == 0 and len_b % 2 == 1) or (
                len_a % 2 == 1 and len_b % 2 == 0
            ):
                max_index -= 1
            elif len_a % 2 == 1 and len_b % 2 == 1:
                max_index -= 2
        elif mode == "trunc":
            order = max(order_a, order_b)
            min_index = order_ab // 2 - order - 1
            if len_a % 2 == 0:
                max_index = order_ab // 2 + order - 2
            else:
                max_index = order_ab // 2 + order - 3

        return a_conv_b[:, min_index : max_index + 1]

    elif method == "toeplitz":
        return torch.matmul(toeplitz_matrix(b, len_a), a.permute([1, 0])).permute(
            [1, 0]
        )

    elif method == "torch":
        return torch.nn.functional.conv1d(
            a.view(dim_a[0], 1, dim_a[1]),
            b.flip([1]).view(1, 1, b.size()[1]),
            padding=(len_b - 1) // 2,
        ).view(dim_a[0], dim_a[1])


def conv2d_separable(a, w, mode="trunc", method="toeplitz"):
    """
    Compute 2d two-sided convolution a * w.

    Here a is an of array coefficients and w1, w2 define a "separable" array w.


    Parameters
    ----------
    a: float-valued Pytorch tensor of size [batch_size heigth width]
        2d array of coefficients
    w: list of length 2 containing two float-valued 1d Pytorch tensors
        sequence of coefficients which define the components of the kernel
    mode: str in {full, trunc}, optional
        compute all convolution terms (full) or truncate (trunc) to nonzero-coefficients determined
        by (maximal) length of input

    Returns
    -------
    a_conv_w: float-valued Pytorch tensor of size ()
        nonzero coefficients of two-sided convolution a * w
    """
    # Initialization
    batch_size, num_rows, num_cols = a.size()
    switch_axes = [1, 0]

    # Perform 1d convolution in each spatial direction
    y = conv1d(a.view(batch_size * num_rows, num_cols), w[1], mode=mode, method=method)
    y = conv1d(
        y.permute(switch_axes).reshape(batch_size * num_cols, num_rows),
        w[0],
        mode=mode,
        method=method,
    )

    # Reshape back to batched format
    return (
        y.reshape(num_cols, batch_size * num_rows)
        .permute(switch_axes)
        .view(batch_size, num_rows, num_cols)
    )


def gaussian_kernel(sigma, threshold=1e-03):
    """
    Sample truncated Gaussian pdf on grid.

    Parameters
    ----------
    sigma: float
        standard deviation of Gaussian
    threshold: float
        sample Gaussian on domain centered at zero containing (1-threshold) of probability mass

    Returns
    -------
    kernel: float-valued Pytorch tensor of size (1, size)
        values of Gaussian sampled on uniformly spaced integer-valued grid of "support"
    """
    # Construct equispaced grid (with spacing one)
    support_radius = int(np.ceil(norm.ppf(1 - threshold / 2, loc=0, scale=sigma)))
    support_right_intval = torch.arange(1, support_radius + 1).float()

    # Evaluate Gaussian
    kernel = torch.zeros(2 * support_radius + 1)
    kernel[support_radius] = 1 / (np.sqrt(2 * np.pi) * sigma)
    kernel[support_radius + 1 :] = kernel[support_radius] * torch.exp(
        -0.5 * (support_right_intval / sigma) ** 2
    )
    kernel[0:support_radius] = kernel[
        torch.arange(2 * support_radius, support_radius, -1)
    ]

    # Discretized normalization: ensure kernel integrates to one
    return kernel.view(1, 2 * support_radius + 1) / torch.sum(kernel)
