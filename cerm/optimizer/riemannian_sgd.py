"""Custom optimizer constrained parameters: Riemannian gradient descent"""

from typing import List

import torch
from torch import Tensor

from cerm.constraints.manifold import ConstrainedManifold
from cerm.network.constrained_params import ConstrainedParameter


class RSGD(torch.optim.Optimizer):

    """Custom optimizer for Riemannian Stochastic Gradient Descent."""

    def __init__(
        self,
        params: List[ConstrainedParameter],
        lr: float = 1e-03,
        weight_decay: float = 0.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        params: List[Tensor]
            iterable containing constrained parameters to be updated
        manifolds: List[ConstrainedManifold]
            manifolds associated to constrained parameters
        lr: float
            learning rate
        weight_decay: float
            weight decay factor
        """
        if lr < 0.0:
            raise ValueError(f"Given learning rate {lr} should be positive")
        if weight_decay < 0.0:
            raise ValueError(f"Given weight decay {weight_decay} should be positive")

        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    # @_use_grad_for_differentiable
    def step(self, closure=None) -> None:
        """Take gradient descent step using Riemannian SGD."""
        assert closure == None, "Closures are not yet supported"

        for group in self.param_groups:
            for params in group["params"]:
                # todo: set state (is this necessary?)
                current_state = self.state[params]
                if len(current_state) == 0:
                    current_state["step"] = 0

                manifold = params.constrained_manifold

                # Compute gradient on riemannian submanifold
                if len(params.grad) > 0:
                    jac_graph_map, grad_manifold = manifold.gradient(
                        params,
                    )
                else:
                    raise ValueError("Gradients for parameters have not been computed")

                # Take a descent step in the chart
                params.data[
                    manifold.vars_and_coords.coords_arr["params_dim"],
                    manifold.vars_and_coords.coords_arr["row_dim"],
                ] -= (
                    group["lr"] * grad_manifold.flatten()
                )

                # Evaluate inverse chart using first order Taylor approximation
                # Todo: move to constrained manifold module
                params.data[
                    manifold.vars_and_coords.vars_arr["params_dim"],
                    manifold.vars_and_coords.vars_arr["row_dim"],
                ] -= (
                    group["lr"] * (jac_graph_map @ grad_manifold).flatten()
                )

                # Refine using Newton
                new_params, groups_not_converged, _ = manifold.refine_point(params.clone())
                if not groups_not_converged:
                    params.copy_(new_params)

                current_state["step"] += 1
