"""Contains custom objects and tuples."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


class VarsAndCoords:
    """Class to split input to zero map into variables and graph coordinates."""

    def __init__(
        self,
        vars_per_group: List[np.ndarray],
        vars_arr: Dict[str, Tensor],
        vars_jac: Dict[str, Tensor],
        vars_jac_inv_chart: Dict[str, Tensor],
        coords_per_group: List[np.ndarray],
        coords_arr: Dict[str, Tensor],
        coords_jac: Dict[str, Tensor],
        coords_jac_inv_chart: Dict[str, Tensor],
        dim_in: int,
        dim_manifold: int,
        num_equations: int,
    ) -> None:
        """Initialize indices defining a split of the input into variables and coordinates. In the
        context of the manifold we interpret the vars (variables) as points on the manifold and
        coords (coordinates) as the associated coordinates in a (graph) chart. This interpretation
        arises from the implicit function theorem.

        Parameters
        ----------
        vars_per_group: List[np.ndarray]
            indices interpreted as variables; used to access variables in 1d array of shape
            [dim_in] (i.e. per group)
        vars_arr: Dict[str, Tensor]
            indices interpreted as variables; used to access variables in input of shape
            [num_groups dim_in] (simultanuous acces to all groups)
        vars_jac: Dict[str, Tensor]
            column indices of jacobian zero map interpreted as variables; used to access array
            of shape [num_groups num_eqs dim_in] (simultanuous acces to all groups)
        coords_per_group: List[np.ndarray]
            indices interpreted as coordinates; used to access coords in in 1d array of shape
            [dim_in] (i.e. per group)
        coords_arr: Dict[str, Tensor]
            params indices interpreted as coordinates on manifold; used to access variables in
            input of shape [num_groups dim_in] (simultanuous acces to all groups)
        coords_jac: Dict[str, Tensor]
            column indices of jacobian zero map interpreted as coordinates ; used to access array
            of shape [num_groups num_eqs dim_in] (simultanuous acces to all groups)
        dim_in: int
            input dimension (number of free parameters)
        dim_manifold: int
            dimension of manifold
        num_equations: int
            number of equations (constraints)
        """
        # Initialize dimensions
        self.num_equations = num_equations
        self.dim_manifold = dim_manifold
        self.dim_in = dim_in

        # Initialize indices to access variables
        self.vars_per_group = vars_per_group
        self.vars_arr = vars_arr
        self.vars_jac = vars_jac
        self.vars_jac_inv_chart = vars_jac_inv_chart

        # Initialize indices to access coordinates
        self.coords_per_group = coords_per_group
        self.coords_arr = coords_arr
        self.coords_jac = coords_jac
        self.coords_jac_inv_chart = coords_jac_inv_chart

    def update(self, params_list: List[int]) -> None:
        """Choose new coordinates and variables for prescribed params.

        Parameters
        ----------
        params_idx: List[int]
            list of params indices for which coordinates and variables are to be updated
        """
        # Goal -> the inputted params have improperly chosen variables and coords -> reshuffle
        num_params_groups = len(params_list)

        # Create new vars+coords:
        reshuffled_vars_coords = init_vars_and_coords(
            self.dim_in, self.dim_manifold, self.num_equations, num_params_groups
        )

        stepsize_vars = self.num_equations
        stepsize_vars_jac = self.num_equations**2
        stepsize_vars_jac_inv_chart = self.num_equations * self.dim_manifold

        stepsize_coords = self.dim_manifold
        stepsize_coords_jac = self.num_equations * self.dim_manifold
        stepsize_coords_jac_inv_chart = self.dim_manifold**2

        # update all the params indexes to the new ones from reshuffled:
        for idx, params_idx in enumerate(params_list):
            # Update list of chosen variables and coordinates
            self.vars_per_group[params_idx] = reshuffled_vars_coords.vars_per_group[idx]
            self.coords_per_group[params_idx] = reshuffled_vars_coords.coords_per_group[idx]

            # Variables and coordinates params
            self.vars_arr["row_dim"][
                params_idx * stepsize_vars : (params_idx + 1) * stepsize_vars
            ] = reshuffled_vars_coords.vars_arr["row_dim"][
                idx * stepsize_vars : (idx + 1) * stepsize_vars
            ]
            self.coords_arr["row_dim"][
                params_idx * stepsize_coords : (params_idx + 1) * stepsize_coords
            ] = reshuffled_vars_coords.coords_arr["row_dim"][
                idx * stepsize_coords : (idx + 1) * stepsize_coords
            ]

            # Variables and coordinates jacobian zero map
            self.vars_jac["col_dim"][
                params_idx * stepsize_vars_jac : (params_idx + 1) * stepsize_vars_jac
            ] = reshuffled_vars_coords.vars_jac["col_dim"][
                idx * stepsize_vars_jac : (idx + 1) * stepsize_vars_jac
            ]
            self.coords_jac["col_dim"][
                params_idx * stepsize_coords_jac : (params_idx + 1) * stepsize_coords_jac
            ] = reshuffled_vars_coords.coords_jac["col_dim"][
                idx * stepsize_coords_jac : (idx + 1) * stepsize_coords_jac
            ]

            # Coordinates jacobian inverse chart
            self.coords_jac_inv_chart["row_dim"][
                params_idx
                * stepsize_coords_jac_inv_chart : (params_idx + 1)
                * stepsize_coords_jac_inv_chart
            ] = reshuffled_vars_coords.coords_jac_inv_chart["row_dim"][
                idx * stepsize_coords_jac_inv_chart : (idx + 1) * stepsize_coords_jac_inv_chart
            ]

            # Variables jacobian inverse chart
            self.vars_jac_inv_chart["row_dim"][
                params_idx
                * stepsize_vars_jac_inv_chart : (params_idx + 1)
                * stepsize_vars_jac_inv_chart
            ] = reshuffled_vars_coords.vars_jac_inv_chart["row_dim"][
                idx * stepsize_vars_jac_inv_chart : (idx + 1) * stepsize_vars_jac_inv_chart
            ]


def init_vars_and_coords(
    dim_in: int,
    dim_manifold: int,
    num_equations: int,
    num_params_groups: int,
) -> VarsAndCoords:
    """Choose coordinates (columns) randomly to define a chart.

    Parameters
    ----------
    dim_in: int
        input dimension (number of free parameters)
    dim_manifold: int
        dimension of manifold
    num_equations: int
        number of equations (constraints)
    num_params_groups: int
        number of params

    Returns
    -------
    vars_and_coords: VarsAndCoords
        wrapper around indices providing access to variables and coordinates
    """
    # Select which indices will be interpreted as inverse charts (variables)
    # There are num_equations of these (per params)
    vars_per_group = [
        np.random.choice(dim_in, num_equations, replace=False).tolist()
        for _ in range(num_params_groups)
    ]

    # Select which indices will be interpreted as coordinates
    coords_per_group = [[idx for idx in range(dim_in) if idx not in var] for var in vars_per_group]

    # Next, we construct indices to split up the jacobian

    # For the column indices we repeat the variables, as chosen above, num_equations times because
    # we want to extract these variables from the jacobian for each row.
    vars_jac = {}
    vars_jac["col_dim"] = torch.tensor(vars_per_group).repeat(1, num_equations).flatten()

    # To create the row indices, we first arange num_equations since that is the number of rows.
    # Afterwards, we repeat it num_variables=num_equations times since we want a row index for each
    # of the column indices we just found above.
    vars_jac["row_dim"] = (
        torch.arange(num_equations)  #
        .view(num_equations, 1)
        .repeat(1, num_equations)
        .flatten()
        .repeat(num_params_groups)
    )

    # For the params indices, we need to arange first num_params_groups times, since that is how
    # many params we have. Since we need them for all variables and equations, we repeat this
    # num_equations * num_equations times (num_variables=num_equations)
    vars_jac["params_dim"] = (
        torch.arange(num_params_groups)
        .view(num_params_groups, 1)
        .repeat(1, num_equations**2)
        .flatten()
    )

    # Indices jacobian interpreted as coordinates on wavelet manifolds -- same as above, but then
    # for the coordinates
    coords_jac = {}
    coords_jac["col_dim"] = torch.tensor(coords_per_group).repeat(1, num_equations).flatten()

    coords_jac["row_dim"] = (
        torch.arange(num_equations)
        .view(num_equations, 1)
        .repeat(1, dim_manifold)
        .flatten()
        .repeat(num_params_groups)
    )

    coords_jac["params_dim"] = (
        torch.arange(num_params_groups)
        .view(num_params_groups, 1)
        .repeat(1, num_equations * dim_manifold)
        .flatten()
    )

    # Indices params interpreted as variables in zero-finding problem
    vars_arr = {}
    vars_arr["row_dim"] = torch.tensor(vars_per_group).flatten()
    vars_arr["params_dim"] = (
        torch.arange(num_params_groups)
        .view(num_params_groups, 1)
        .repeat(1, num_equations)
        .flatten()
    )

    coords_arr = {}
    coords_arr["row_dim"] = torch.tensor(coords_per_group).flatten()
    coords_arr["params_dim"] = (
        torch.arange(num_params_groups)
        .view(num_params_groups, 1)
        .repeat(1, dim_manifold)
        .flatten()
    )

    # Indices jacobian inverse chart interpreted as coordinates in zero-finding problem
    coords_jac_inv_chart = {}
    coords_jac_inv_chart["col_dim"] = (
        torch.arange(dim_manifold)
        .view(1, 1, dim_manifold)
        .repeat(num_params_groups, dim_manifold, 1)
        .flatten()
    )
    coords_jac_inv_chart["row_dim"] = (
        torch.tensor(coords_per_group).flatten().unsqueeze(-1).repeat(1, dim_manifold).flatten()
    )
    coords_jac_inv_chart["params_dim"] = (
        torch.arange(num_params_groups)
        .view(num_params_groups, 1)
        .repeat(1, dim_manifold**2)
        .flatten()
    )

    # Indices jacobian inverse chart interpreted as variables in zero-finding problem
    vars_jac_inv_chart = {}
    vars_jac_inv_chart["col_dim"] = (
        torch.arange(dim_manifold)
        .view(1, 1, dim_manifold)
        .repeat(num_params_groups, num_equations, 1)
        .flatten()
    )
    vars_jac_inv_chart["row_dim"] = (
        torch.tensor(vars_per_group).flatten().unsqueeze(-1).repeat(1, dim_manifold).flatten()
    )
    vars_jac_inv_chart["params_dim"] = (
        torch.arange(num_params_groups)
        .view(num_params_groups, 1)
        .repeat(1, num_equations * dim_manifold)
        .flatten()
    )

    var_and_coords = VarsAndCoords(
        vars_per_group,
        vars_arr,
        vars_jac,
        vars_jac_inv_chart,
        coords_per_group,
        coords_arr,
        coords_jac,
        coords_jac_inv_chart,
        dim_in,
        dim_manifold,
        num_equations,
    )

    return var_and_coords
