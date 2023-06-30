"""Tools for computing points and gradients on the wavelet manifold."""

import logging
import numpy as np
import torch

from typing import Tuple, Union, List
from torch import Tensor

from cerm.constraints.constraints import Constraint
from cerm.examples.wavelets.circular_conv import circular_conv

# Module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def regularity_equations(
    order: int, moments: int = 1, device: str = "cpu"
) -> Tuple[Tensor, Tensor]:
    """
    Construct linear operator associated to H'(1/2) = 0

    Parameters
    ----------
    order: List[int, int]
        order of lpf
    moments: int
        number of moments to vanish
    device: str
        device on which operators are stored

    Returns
    ------
    operator_order_root: float-valued PyTorch tensor of size [2m-1]
        linear operator associated to H'(1/2) = 0
    """
    operator_order_root = torch.arange(1 - order, order).float().to(device)

    if order % 2 == 1:
        operator_order_root[slice(1, 2 * order - 2, 2)] *= -1
    else:
        operator_order_root[slice(0, 2 * order - 1, 2)] *= -1

    return operator_order_root


class QMFConstraint(Constraint):

    """Implementation quadratic mirror filter (QMF) constraints"""

    def __init__(self, num_filters: int, order: int) -> None:
        """
        Initialize dimension and number of low pass filters

        Parameters
        ----------
        num_filters: int
            number of filters
        order: int
            order of each filter
        """
        self.order = order
        self.num_spatial_dim = 1
        num_eqs = order + 1
        dim_filter = 2 * order - 1

        super(QMFConstraint, self).__init__(dim_filter, num_eqs, num_filters)

    def __call__(self, lpf: Tensor) -> Tensor:
        """
        Evaluate QMF conditions

        Parameters
        ----------
        lpf: float-valued PyTorch tensor of size [num_filters 2 * order - 1]
            low pass filters

        Returns
        -------
        float-valued PyTorch tensor of size [num_filters order + 1]
            zero finding map F evaluated at low pass filter
        """
        # Compute h * h_flip
        lpf_conv_lpf_flip = circular_conv(
            lpf, torch.flip(lpf, dims=[-1]), self.num_spatial_dim
        )

        # Extract positive even indices
        even_idx = slice(2 * (self.order - 1), 4 * (self.order - 1) + 1, 2)

        # Orthonormality constraints
        fval = lpf_conv_lpf_flip[..., even_idx]
        fval[..., 0] = fval[..., 0] - 1

        # Equation non-degeneracy: H(0) = 1
        nondeg_equation = torch.sum(lpf, dim=[-1]) - np.sqrt(2)

        return torch.cat(
            (
                fval,
                nondeg_equation.unsqueeze(-1),
            ),
            dim=-1,
        )
