"""Example constraint Stiefel manifold."""

import torch
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

        super().__init__(num_rows * num_cols, num_eqs, num_groups)

    @staticmethod
    def _delta(i: int, j: int) -> int:
        """Kronecker delta.

        Parameters
        ----------
        int: i
            index
        int j:
            index

        Returns
        -------
        int
            kronecker delta of i and j
        """
        return 1 if i == j else 0

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
        eqs = []

        for i in range(self.num_cols):
            col = mat[:, :, i]
            for j in range(i, self.num_cols):
                eqs.append(torch.sum(mat[:, :, j] * col, dim=-1) - self._delta(i, j))

        eqs = torch.cat(eqs, dim=-1)

        if self.num_groups == 1:
            return eqs.unsqueeze(0)
        else:
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
