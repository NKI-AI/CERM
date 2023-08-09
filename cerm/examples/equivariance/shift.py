"""Example circular shift-equivariance."""

import torch
import numpy as np

from torch import Tensor
from typing import Tuple

from cerm.constraints.constraints import Constraint
from cerm.network.constrained_params import ConstrainedParameter


def init_dft_matrices(dim: int) -> Tensor:
    """Construct matrix representation DFT and its inverse.

    Parameters
    ----------
    dim: int
        dimension DFT-matrix

    Returns
    -------
    dft: complex-valued PyTorch tensor of shape [dim dim]
        DFT matrix
    inv_dft: complex-valued PyTorch tensor of shape [dim dim]
        inverse DFT matrix
    """
    k = torch.arange(dim) + 0j
    dft = torch.exp(-1j * 2 * np.pi * k.view(dim, 1) @ k.view(1, dim) / dim)
    inv_dft = torch.conj(torch.transpose(dft, 0, 1)) / dim
    return dft, inv_dft


def params_to_matrix(params: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
    """Convert flattened parameters to real and imaginary part weight matrix.

    Parameters
    ----------
    params: float-valued PyTorch tensor of shape [num_groups 2 * dim^2]
        flattened view of real and imaginary part weight matrix
    dim: int
        dimension of (square) weight matrix

    Returns
    -------
    real_mat: float-valued PyTorch tensor of shape [num_groups dim dim]
        real part of weight matrix
    imag_mat: float-valued PyTorch tensor of shape [num_groups dim dim]
        imaginary part of weight matrix
    """
    num_groups = params.shape[0]
    real_mat = params[:, 0 : dim**2].view(num_groups, dim, dim)
    imag_mat = params[:, dim**2 :].view(num_groups, dim, dim)
    return real_mat, imag_mat


class ShiftConstraint(Constraint):
    """Constraint shift equivariant layer."""

    def __init__(self, num_groups: int, dim: int) -> None:
        """Initialize shift equivariant constraint.

        Parameters
        ----------
        dim: int
            dimension linear layer
        num_groups: int
            number of parameters groups
        """
        self.dim = dim

        # DFT matrix and its inverse
        dft, inv_dft = init_dft_matrices(self.dim)
        self.dft = dft.unsqueeze(0).repeat(num_groups, 1, 1)
        self.inv_dft = inv_dft.unsqueeze(0).repeat(num_groups, 1, 1)

        # Non-diagonal indices (linear indexing in row major order)
        diag_idx = [j * (self.dim + 1) for j in range(self.dim)]
        self.non_diag_idx = [j for j in range(self.dim**2) if j not in diag_idx]

        num_eqs = 2 * dim * (dim - 1)
        super().__init__(2 * dim**2, num_eqs, num_groups)

    def __call__(self, params: Tensor) -> Tensor:
        """Evaluate constraint map.

        Parameters
        ----------
        params: float-valued PyTorch tensor of shape [num_groups 2 * dim^2]
            flattened view of real and imaginary part weight matrix

        Returns
        -------
        float-valued PyTorch tensor of shape [num_groups num_eqs]
            constraint map evaluated at params
        """
        # Compute y = V^{-1} W V
        real_mat, img_mat = params_to_matrix(params, self.dim)
        mat = real_mat + 1j * img_mat
        y = (self.inv_dft @ mat @ self.dft).flatten(start_dim=1)

        # Drop diagonal terms
        return torch.view_as_real(y[:, self.non_diag_idx]).flatten(start_dim=1)


class ShiftEquivariantLayer(torch.nn.Module):
    """Example shift equivariant linear layer."""

    def __init__(
        self,
        dim: int,
        bias: bool = False,
    ) -> None:
        """Initialize parameters linear layer.

        Parameters
        ----------
        dim: int
            input and output dimension linear layer
        bias: bool
            bias
        """
        super().__init__()

        self.dim = dim
        self.num_groups = 1
        self.params = ConstrainedParameter(
            init_params=self._init_params(),
            constraint=ShiftConstraint(self.num_groups, self.dim),
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.rand(self.dim, 2))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(self.dim, 2))

    def _init_params(self) -> Tensor:
        """Initialize random shift-equivariant weight matrix: W = V diag(x) V^{-1}

        Returns
        -------
        float-valued PyTorch tensor of shape [1 2*dim^2]
            randomly initialized shift-equivariant weight matrix (flattened)
        """
        dft, inv_dft = init_dft_matrices(self.dim)
        diag_elements = torch.normal(torch.zeros(self.dim), 1 / self.dim**1.5)
        mat = dft @ torch.diag(diag_elements + 0j) @ inv_dft
        return torch.stack((torch.real(mat), torch.imag(mat))).flatten().unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate shift-equivariant layer.

        Parameters
        ----------
        x: float-valued PyTorch Tensor of shape [batch_size dim]
            input to equivariant layer

        Returns
        -------
        float-valued PyTorch tensor of shape [batch_size dim]
            output equivariant layer
        """
        real_mat, imag_mat = params_to_matrix(self.params, self.dim)
        y = torch.nn.functional.linear(x, real_mat.squeeze(0), bias=self.bias[:, 0])
        return torch.nn.functional.linear(y, imag_mat.squeeze(0), bias=self.bias[:, 1])
