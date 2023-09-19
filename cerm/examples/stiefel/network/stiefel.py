"""Example constraint Stiefel manifold."""

import torch
import numpy as np

from torch import Tensor

from cerm.constraints.constraints import Constraint
from cerm.network.constrained_params import ConstrainedParameter


def sample_skew_symmetric_matrix(dim: int) -> torch.Tensor:
    """Sample random skew symmetric matrix.

    Parameters
    ----------
    dim: int
        dimension matrix

    Returns
    -------
    mat: float-valued PyTorch Tensor of shape [dim dim]
        randomly sampled skew-symmetric matrix
    """
    num_params = ((dim - 1) * dim) // 2
    params = torch.rand(num_params)
    mat = torch.zeros(dim, dim)

    idx_0 = 0
    for j in range(dim - 1):
        # Extract elements for j-th row and and j-th column
        num_elements = dim - j - 1
        idx_1 = idx_0 + num_elements
        elements = params[idx_0:idx_1]

        # Construct skew-symmetric row-column pair
        mat[j, j + 1 :] = elements
        mat[j + 1 :, j] = -elements

        idx_0 = idx_1

    return mat


def sample_orthogonal_matrix(dim: int) -> torch.Tensor:
    """Sample random orthogonal matrix.

    Parameters
    ----------
    dim: int
        dimension matrix

    Returns
    -------
    float-valued PyTorch Tensor of shape [dim dim]
        randomly sampled orthogonal matrix
    """
    return torch.linalg.matrix_exp(sample_skew_symmetric_matrix(dim))


class StiefelConstraint(Constraint):
    """Example of constraining parameters to Stiefel manifold"""

    def __init__(self, num_groups: int, num_rows: int, num_cols: int) -> None:
        """Initialize parameters spherical constraints.

        Parameters
        ----------
        num_groups: int
            number of groups
        num_rows: int
            number of rows
        num_cols:
            number of columns
        """
        assert (
            num_rows >= num_cols
        ), "Number of rows should be larger than number of columns"
        num_eqs = (num_cols * (num_cols + 1)) // 2

        self.num_rows = num_rows
        self.num_cols = num_cols

        # Indices upper triangular part of square matrix
        idx_left = []
        idx_right = []
        for row_idx in range(self.num_cols):
            idx_left += [row_idx for _ in range(num_cols - row_idx)]
            idx_right += [k for k in range(row_idx, self.num_cols)]
        self.mat_upper_idx = [idx_left, idx_right]

        self.eye = torch.eye(self.num_cols)

        super().__init__(num_rows * num_cols, num_eqs, num_groups)

    def __call__(self, params: Tensor) -> Tensor:
        """Zero map associated stiefel manifold.

        Parameters
        ----------
        params: float-valued PyTorch tensor of shape [num_groups dim_params]
            flattened matrix to be constrained

        Returns
        -------
        float-valued PyTorch tensor of shape [num_groups num_eqs]
            evaluated stiefel zero map
        """
        mat = params.view(self.num_groups, self.num_rows, self.num_cols)
        orth_constraint = torch.transpose(mat, 1, 2) @ mat - self.eye.to(params.device)
        eqs = orth_constraint[:, self.mat_upper_idx[0], self.mat_upper_idx[1]]

        # Memory efficient implementation
        # for i in range(self.num_cols):
        #     col = mat[:, :, i]
        #     for j in range(i, self.num_cols):
        #        eqs.append(torch.sum(mat[:, :, j] * col, dim=-1) - self._delta(i, j))
        return eqs


class StiefelLayer(torch.nn.Module):
    """Example linear layer with weights constrained to stiefel manifold."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
    ) -> None:
        """Initialize parameters Stiefel manifold and bias.

        Parameters
        ----------
        dim_in: int
            input dimension linear layer
        dim_out: int
            output dimension linear layer
        bias: bool
            bias
        """
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_groups = 1

        self.params = ConstrainedParameter(
            init_params=self._init_stiefel_params(),
            constraint=StiefelConstraint(self.num_groups, self.dim_out, self.dim_in),
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.rand(self.dim_out))
        else:
            self.register_parameter("bias", None)

    def _init_stiefel_params(self) -> Tensor:
        """Initialize random stiefel matrix.

        Returns
        -------
        float-valued PyTorch tensor of shape [1 dim_out * dim_in]
            initialize initial point on stiefel manifold
        """
        mat = sample_orthogonal_matrix(self.dim_out)[:, 0 : self.dim_in]
        return mat.flatten().unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Stiefel layer.

        Parameters
        ----------
        x: float-valued PyTorch Tensor of shape [batch_size dim_in]
            input to stiefel layer

        Returns
        -------
        float-valued PyTorch tensor of shape [batch_size dim_out]
            output stiefel layer
        """
        return torch.nn.functional.linear(
            x, self.params.view(self.dim_out, self.dim_in), bias=self.bias
        )


class StiefelConv2d(torch.nn.Module):
    """Convolutional layer with filter constrained to Stiefel manifold."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        padding_mode: str = "zeros",
        filter_batch_size: int = 128,
        device: bool = None,
    ) -> None:
        """Initialize parameters Stiefel manifold and bias.

        Parameters
        ----------
        in_channels: int
            number input channels
        out_channels: int
            number output channels (feature maps)
        kernel_size: int
            size of kernel
        stride: int
            stride in convolution
        padding: int
            amount of padding before taking convolution
        padding_mode: str
            padding method
        param_batch_size: int
            split parameters (filters) up in batches for performance
        bias: bool
            include bias if true
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device

        # Initialize filters constrained to Stiefel manifold
        self.num_filters = self.out_channels * self.in_channels
        self.filter_batch_size = filter_batch_size
        self._init_stiefel_filters()

        # Bias
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    out_channels,
                )
            )
        else:
            self.bias = None

        self._init_bias()

    def _init_bias(self) -> None:
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size**2
            if fan_in != 0:
                bound = 1 / np.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def _init_stiefel_filters(self) -> Tensor:
        """Initialize filters on stiefel manifold.

        Returns
        -------
        float-valued PyTorch tensor of shape [out_channels kernel_size * kernel_size]
            initial filters on stiefel manifold
        """
        num_batches = self.num_filters // self.filter_batch_size
        remainder = self.num_filters % self.filter_batch_size
        batch_sizes = [self.filter_batch_size for _ in range(num_batches)]
        if remainder > 0:
            batch_sizes += [remainder]

        self.kernels = torch.nn.ParameterList([])
        for bsize in batch_sizes:
            orth_mat = torch.stack(
                [sample_orthogonal_matrix(self.kernel_size) for _ in range(bsize)]
            ).flatten(start_dim=1)

            self.kernels.append(
                ConstrainedParameter(
                    init_params=orth_mat,
                    constraint=StiefelConstraint(
                        bsize, self.kernel_size, self.kernel_size
                    ),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        params = torch.concat(
            [
                k.view(k.shape[0], self.kernel_size, self.kernel_size)
                for k in self.kernels
            ]
        )
        kernels = params.view(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        return torch.nn.functional.conv2d(
            x, kernels, bias=self.bias, stride=self.stride, padding=self.padding
        )
