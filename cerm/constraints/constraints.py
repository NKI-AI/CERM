"""Implementation constraints."""

import abc
import logging
from typing import Callable

import torch
from torch import Tensor

# Module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Constraint(abc.ABC):
    """Class providing tools and utilities for handling constraints."""

    def __init__(self, num_params: int, num_eqs: int, num_groups: int) -> None:
        """
        Parameters
        ----------
        num_params: int
            number of input parameters to zero map
        num_eqs: int
            number of equations
        num_groups: int
            number of groups (groups to which we apply the same constraints)
        """
        if num_params <= num_eqs:
            raise ValueError("Number of params should exceed number of equations!")

        # Set dimensions
        self.__num_params = num_params
        self.__num_eqs = num_eqs
        self.__num_groups = num_groups

        # Vectorized computation jacobian
        # self._jac = torch.func.jacrev(self.__call__, argnums=0)

        self.f_per_sample = lambda x: self.__call__(x.unsqueeze(0)).squeeze(0)
        self.jac_per_sample = torch.func.jacrev(self.f_per_sample, argnums=0)
        self._jac = torch.func.vmap(self.jac_per_sample)

    @property
    def num_params(self) -> int:
        """Dimension domain zero map."""
        return self.__num_params

    @property
    def num_eqs(self) -> int:
        """Number of equations."""
        return self.__num_eqs

    @property
    def num_groups(self) -> int:
        """Dimension of output zero map."""
        return self.__num_groups

    @abc.abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate zero map f.

        Parameters
        ----------
        x: float-valued Pytorch tensor of shape [num_groups num_params]
            input to zero map; applies same map to every group of params

        Returns
        -------
        float-valued PyTorch tensor of shape [num_groups num_eqs]
            zero map evaluated at input
        """

    def eval_jac(self, x: Tensor) -> Tensor:
        """Evaluate Jacobian zero map f.

        Parameters
        ----------
        x: float-valued PyTorch tensor of shape [num_groups num_params]
            input to evaluate jacobian at

        Returns
        -------
        float-valued PyTorch tensor of shape [num_groups num_eqs num_params]
            jacobian evaluated at x
        """
        # Note: since the same zero map is applied to every group the Jacobian
        # will be a block diagonal matrix. It is not necessary to compute the
        # zero blocks, but for now we do it anyway to avoid looping over the
        # num_group dimension
        dzero_map = self._jac(x)
        return dzero_map

        # # Discard zero blocks and store Df applied to
        # group_idx = torch.arange(self.num_groups)
        # return dzero_map[group_idx, :, group_idx]
