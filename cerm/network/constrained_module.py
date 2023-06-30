"""Implementation constrained module"""

import torch
import abc

from typing import Tuple, Dict, List

from cerm.constraints.constraints import Constraint
from cerm.constraints.manifold import ConstrainedManifold
from torch import Tensor


class ConstrainedParameter(torch.nn.Parameter):

    """Duplicate torch parameter to distinguish between constrained and unconstrained"""

    def __init__(self, data: Tensor = None, requires_grad: bool = True) -> None:
        super(ConstrainedParameter, self).__init__()
        self.data = data
        self.requires_grad = requires_grad


class ConstrainedModule(abc.ABC, torch.nn.Module):

    """Implementation constrained layer"""

    def __init__(self, constraint: Constraint) -> None:
        """
        Initialize constraint defining the manifold

        Parameters
        ----------
        constraint: Constraint
            constraints defining manifold
        """
        super(ConstrainedModule, self).__init__()
        self.constrained_manifold = ConstrainedManifold(constraint)
        # self.params = torch.nn.Parameter(manifold.sample_point())
        # params = torch.rand(constraint.num_groups, constraint.num_params)
        # params = self.constraint_manifold.refine_point(params)
        # self.params = torch.nn.Parameter(params)


def split_params(
    model: torch.nn.Module,
) -> Tuple[List[Dict[str, ConstrainedParameter]], List[torch.nn.Parameter]]:
    """
    Split parameters of model into constrained and unconstrained part

    Parameters
    ----------
    model: torch.nn.Module
        neural network containing constrained and free parameters

    Returns
    -------
    constrained_params: Dict[str, List[ConstrainedParameter]]
    constrained_manifolds: List[ConstrainedManifold]
        constrained manifolds associated to constrained layers
    free_params: List[torch.nn.Parameter]
        free parameters
    """
    free_params = []
    constrained_params = []
    constrained_manifolds = []

    for layer in model.modules():
        if isinstance(layer, ConstrainedModule):
            # Note: constrained parameters should be gathered in one dict
            # to indicate that they correspond to one constrained manifold
            constrained_manifolds.append(layer.constrained_manifold)
            p_constrained = []
            for name, w in layer.named_parameters():
                if isinstance(w, ConstrainedParameter):
                    p_constrained.append(w)
                else:
                    free_params.append(w)
            # Ensure that found constrained parameters are stored together
            # so that we can associate them to one manifold
            constrained_params.append({"params": p_constrained})
        else:
            for name, w in layer.named_parameters():
                free_params.append(w)

    return free_params, constrained_manifolds, constrained_params
