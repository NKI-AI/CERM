"""Tools for computing points and gradients on the constrained manifold."""

import logging
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from cerm.constraints import coordinate_utils
from cerm.constraints.constraints import Constraint
from cerm.constraints.newton import Newton

# Module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ConstrainedManifold:
    """Class representing a manifold defined by a constraint."""

    def __init__(self, constraint: Constraint) -> None:
        """Initialize constraint defining the manifold.

        Parameters
        ----------
        constraint: Constraint
            constraints defining manifold
        newton: Newton
            newton's method initialized with suitable parameters
        """
        self.constraint = constraint
        self.newton = Newton()

        self.__dim_manifold = self.constraint.num_params - self.constraint.num_eqs
        self.__vars_and_coords = coordinate_utils.init_vars_and_coords(
            self.constraint.num_params,
            self.__dim_manifold,
            self.constraint.num_eqs,
            self.constraint.num_groups,
        )

    @property
    def dim_manifold(self) -> int:
        """Dimension of manifold defined by constraint."""
        return self.__dim_manifold

    @property
    def vars_and_coords(self) -> coordinate_utils.VarsAndCoords:
        """Dimension of manifold defined by constraint."""
        return self.__vars_and_coords

    def refine_point(self, x0: Tensor, max_attempts: int = 64) -> Tensor:
        """Refine approximation of point on manifold changing coordinates if necessary.

        Parameters
        ----------
        x0: float-valued PyTorch tensor of shape [num_groups num_params]
            approximate point on manifold
        max_attempts: int
            maximal number of attempts to change coordinates

        Returns
        -------
        x: float-valued PyTorch tensor of shape [num_groups num_params]
            zeros f(x) = 0 per group
        singular_group_idx: List[int]
            list of groups where Newton failed or zeros are singular
        min_singular_val: float
            smallest singular value jacobian over groups
        """
        num_iter = 0

        x, groups_not_converged, min_singular_val = self.newton(
            x0.clone(), self.constraint, self.__vars_and_coords
        )

        x[groups_not_converged] = x0[groups_not_converged]

        while groups_not_converged and num_iter <= max_attempts:
            logger.info(f"Attempt: {num_iter}")

            self.__vars_and_coords.update(
                groups_not_converged
            )  # TODO: fix this -> after this step all filters seem to fail

            x, groups_not_converged, min_singular_val = self.newton(
                x, self.constraint, self.__vars_and_coords
            )

            x[groups_not_converged] = x0[groups_not_converged]

            num_iter += 1

        return x, groups_not_converged, min_singular_val

    def eval_jac_inv_chart(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate derivative of parameterization manifold (inverse chart).

        Parameters
        ----------
        x: float-valued PyTorch tensor of shape [num_groups num_params]
            pointin coordinate chart to evaluate jacobian at

        Returns
        -------
        jac_graph_map: float-valued Pytorch Tensor of size [num_groups num_equations dim_manifold]
            derivative of graph map
        jac_inv_chart: float-valued Pytorch Tensor of size [num_groups num_params dim_manifold]
            derivative of inverse chart
        """
        # Dimensions and ordering
        num_params = self.constraint.num_params
        num_equations = self.constraint.num_eqs
        dim_manifold = self.dim_manifold
        num_groups = self.constraint.num_groups
        vars_and_coords = self.vars_and_coords

        # Compute derivative of graph map tilde c
        jac_zero_map = self.constraint.eval_jac(x)
        jac_graph_map = torch.linalg.solve(
            jac_zero_map[
                vars_and_coords.vars_jac["params_dim"],
                vars_and_coords.vars_jac["row_dim"],
                vars_and_coords.vars_jac["col_dim"],
            ].view(num_groups, num_equations, num_equations),
            -jac_zero_map[
                vars_and_coords.coords_jac["params_dim"],
                vars_and_coords.coords_jac["row_dim"],
                vars_and_coords.coords_jac["col_dim"],
            ].view(num_groups, num_equations, dim_manifold),
        )

        # Compute derivative inverse chart
        jac_inv_chart = torch.zeros(num_groups, num_params, dim_manifold).to(
            jac_zero_map.device
        )

        jac_inv_chart[
            vars_and_coords.vars_jac_inv_chart["params_dim"],
            vars_and_coords.vars_jac_inv_chart["row_dim"],
            vars_and_coords.vars_jac_inv_chart["col_dim"],
        ] = jac_graph_map.flatten()

        jac_inv_chart[
            vars_and_coords.coords_jac_inv_chart["params_dim"],
            vars_and_coords.coords_jac_inv_chart["row_dim"],
            vars_and_coords.coords_jac_inv_chart["col_dim"],
        ] = (
            torch.eye(dim_manifold)
            .to(jac_zero_map.device)
            .unsqueeze(0)
            .repeat(num_groups, 1, 1)
            .flatten()
        )

        return jac_graph_map, jac_inv_chart

    @staticmethod
    def riemann_metric(jac_inv_chart: Tensor) -> Tensor:
        """Evaluate riemannian metric at point on manifold.

        Parameters
        ----------
        jac_inv_chart: float-valued Pytorch Tensor of size [num_groups num_params dim_manifold]
            derivative of inverse chart

        Returns
        -------
        metric: float-valued PyTorch tensor of size [num_groups dim_manifold dim_manifold]
            riemannian metric evaluated at point on manifold
        """
        # Note: only need to compute and store upper diagonal part
        return torch.transpose(jac_inv_chart, 1, 2) @ jac_inv_chart

    def gradient(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute gradient of objective on constrained manifold.

        Parameters
        ----------
        x: float-valued PyTorch tensor of shape [num_groups num_params]
            point contained in manifold to evaluate gradient at
        constraint: Constraint
            constraints defining manifold

        Returns
        -------
        jac_graph_map: float-valued Pytorch Tensor of size [num_groups num_equations dim_manifold]
            derivative of graph map
        grad_manifold: float-valued PyTorch tensor of size [num_groups dim_manifold 1]
            gradient on constrained manifold
        """
        num_params = self.constraint.num_params
        num_groups = self.constraint.num_groups

        # Components "uncontrained" derivatives on wavelet manifold
        jac_fun_filter = x.grad.view(num_groups, 1, num_params)

        # Gradient on manifold
        with torch.no_grad():
            #  D(\tilde c)(lambda*) and Dc(lambda*)
            jac_graph_map, jac_inv_chart = self.eval_jac_inv_chart(x)

            # g_{M}
            metric = self.riemann_metric(jac_inv_chart)

            # dL / dx^i(lambda*)
            jac_fun_manifold = jac_fun_filter @ jac_inv_chart

            # grad_L on W_m
            grad_manifold = torch.linalg.solve(metric, jac_fun_manifold.transpose(1, 2))

        return jac_graph_map, grad_manifold
