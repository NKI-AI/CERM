"""Implementation constrained module."""

import abc
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from cerm.constraints.constraints import Constraint
from cerm.constraints.manifold import ConstrainedManifold


class ConstrainedParameter(torch.nn.Parameter):
    """Duplicate torch parameter to distinguish between constrained and unconstrained."""

    def __new__(
        cls,
        init_params: Tensor = None,
        requires_grad: bool = True,
        constraint: Constraint = None,
    ):
        """Add functionality to torch parameter making it into a constrained parameter.

        Parameters
        ----------
        init_params: float-valued PyTorch tensor of shape [num_groups dim_params]
            initial guess for parameters
        requires_grad: bool
            track gradients is true
        constraint: Constraint
            constraint on parameters

        Returns
        -------
        obj: ConstrainedParameter
            reference to instance of ContrainedParameter
        """
        if constraint is None:
            raise ValueError("No constraint was provided")

        if init_params is None:
            # TODO: user should provide a custom initializer this is problem specific
            init_params = torch.rand(constraint.num_groups, constraint.num_params)

        constrained_manifold = ConstrainedManifold(constraint)
        params, _, _ = constrained_manifold.refine_point(init_params)

        obj = super().__new__(cls, data=params, requires_grad=requires_grad)
        obj.constrained_manifold = constrained_manifold

        return obj


def split_params(
    model: torch.nn.Module,
) -> Tuple[List[torch.nn.Parameter], List[ConstrainedParameter]]:
    """Split parameters of model into constrained and unconstrained part.

    Parameters
    ----------
    model: torch.nn.Module
        neural network containing constrained and free parameters

    Returns
    -------
    free_params: List[torch.nn.Parameter]
        free parameters
    constrained_params: List[ConstrainedParameter]
        constrained parameters
    """
    free_params = []
    constrained_params = []

    for name, params in model.named_parameters():
        if isinstance(params, ConstrainedParameter):
            constrained_params.append(params)
        elif isinstance(params, torch.nn.Parameter):
            free_params.append(params)

    return free_params, constrained_params
