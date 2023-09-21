"""Implementation Newton's method to find points on constraint manifold."""

import logging
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from cerm.constraints.constraints import Constraint
from cerm.constraints.coordinate_utils import VarsAndCoords

# Module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Newton:
    """Newton's method for solving overdetermined system of equations."""

    def __init__(
        self,
        max_iter: int = 64,
        tol_zero: float = 1e-05,
        tol_svd: float = 1e-04,
        group_criterium: str = "max",
        param_criterium: str = "l1",
    ) -> None:
        """Use Newton's method to find a solution of f(x) = 0.

        Parameters
        ----------
        max_iter: int
            maximum number of iterations before stopping
        tol_zero: float
            if ||f(x)||_1 > tol_zero, then x is not a proper zero
        tol_svd: float:
            if the smallest singular value of Df(x) is below tol_svd, then x is singular
        group_criterium: str
            criterium used to measure quality zero over all groups
        param_criterium: str
            criterium used to measure quality zero of an individual group
        """
        self.max_iter = max_iter
        self.tol_zero = tol_zero
        self.tol_svd = tol_svd

        # Determine criterium to access quality of zero
        self.group_criterium = group_criterium
        self.param_criterium = param_criterium
        self.eval_group_error = self._get_criterium(self.group_criterium, dim=0)
        self.eval_param_error = self._get_criterium(self.param_criterium, dim=1)
        self.eval_error = lambda y: self.eval_group_error(self.eval_param_error(y))

    @staticmethod
    def _get_criterium(criterium: str, dim: int) -> Callable:
        """Determine criterium to assess quality zero.

        Parameters
        ----------
        criterium: string
            criterium to be used
        dim: int
            specifies over which dimension (group or param) the criterium is applied

        Returns
        -------
        Callable
            anonymous implementation prescribed criterium
        """
        if criterium == "max":
            return lambda y: torch.max(torch.abs(y), dim=dim)[0]
        elif criterium == "l1":
            return lambda y: torch.sum(torch.abs(y), dim=dim)
        elif criterium == "l2":
            return lambda y: torch.sum(y**2, dim=dim)
        else:
            raise NotImplementedError(
                f"Prescribed criterium {criterium} does not exist"
            )

    def _singular_groups(
        self, x: Tensor, constraint: Constraint, vars_and_coords: VarsAndCoords
    ) -> List[int]:
        """Find indices of parameter groups that not converged or are singular.

        Parameters
        ----------
        x0: float-valued PyTorch tensor of shape [num_groups num_params]
            initial guess for zero
        constraint: Constraint

        Returns
        -------
        List[int]
            list of groups where Newton failed or zeros are singular
        min_singular_val: float
            smallest singular value jacobian over groups
        """
        # Check for which parameter groups Newton not converged
        error_per_group = self.eval_param_error(constraint(x))
        error_per_group_inf = torch.isinf(error_per_group)
        error_per_group_nan = torch.isnan(error_per_group)
        inf_idx = torch.where(error_per_group_inf == True)[0]
        nan_idx = torch.where(error_per_group_nan == True)[0]

        # Check for which parameter groups Jacobian are singular
        dfx = constraint.eval_jac(x)[
            vars_and_coords.vars_jac["params_dim"],
            vars_and_coords.vars_jac["row_dim"],
            vars_and_coords.vars_jac["col_dim"],
        ].view(constraint.num_groups, constraint.num_eqs, constraint.num_eqs)
        dfx[inf_idx] = 0
        dfx[nan_idx] = 0

        min_singular_val, _ = torch.min(torch.linalg.svdvals(dfx), dim=1)

        not_converged_or_singular = (
            error_per_group_inf
            | error_per_group_nan
            | (error_per_group > self.tol_zero)
            | (min_singular_val < self.tol_svd)
        )

        return (
            torch.where(not_converged_or_singular == True)[0].tolist(),
            min_singular_val,
        )

    def __call__(
        self, x0: Tensor, constraint: Constraint, vars_and_coords: VarsAndCoords
    ) -> Tensor:
        """Find zero in neighborhood of initial guess.

        Parameters
        ----------
        x0: float-valued PyTorch tensor of shape [num_groups num_params]
            initial guess for zero
        constraint: Constraint

        Returns
        -------
        x: float-valued PyTorch tensor of shape [num_groups num_params]
            zeros f(x) = 0 per group
        singular_group_idx: List[int]
            list of groups where Newton failed or zeros are singular
        min_singular_val: float
            smallest singular value jacobian over groups
        """
        num_groups = constraint.num_groups
        num_eqs = constraint.num_eqs

        with torch.no_grad():
            # Initialization
            num_iter = 0
            x = x0.clone()
            fx = constraint(x)

            # Newton iterations
            while self.eval_error(fx) > self.tol_zero and num_iter <= self.max_iter:
                dfx = constraint.eval_jac(x)[
                    vars_and_coords.vars_jac["params_dim"],
                    vars_and_coords.vars_jac["row_dim"],
                    vars_and_coords.vars_jac["col_dim"],
                ].view(num_groups, num_eqs, num_eqs)

                try:
                    solved_units = (
                        torch.linalg.solve(dfx, fx.unsqueeze(-1)).squeeze(-1).flatten()
                    )
                    x[
                        vars_and_coords.vars_arr["params_dim"],
                        vars_and_coords.vars_arr["row_dim"],
                    ] -= solved_units
                except Exception as exc:
                    logger.exception(exc)
                    break

                # Update error
                fx = constraint(x)
                num_iter += 1

            # Find groups for which Newton failed
            singular_group_idx, min_singular_val = self._singular_groups(
                x, constraint, vars_and_coords
            )
            curr_error = self.eval_error(fx)

            if singular_group_idx:
                logger.warning("Singular parameter groups detected")
                if curr_error < self.tol_zero:
                    logger.info(
                        f"Newton converged to singular point: error = {curr_error}"
                    )
                else:
                    logger.info(f"Newton did not converge: error = {curr_error}")
            else:
                pass
                # logger.info(f"Newton converged. Error = {curr_error}")

            return x, singular_group_idx, min_singular_val
