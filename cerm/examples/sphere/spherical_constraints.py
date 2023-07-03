"""Minimal toy example: linear layer with rows constrained to sphere"""

import torch

from torch import Tensor

from cerm.constraints.constraints import Constraint
from cerm.network.constrained_params import ConstrainedParameter


class SphericalConstraint(Constraint):

    """Simple example of constraining parameters to the unit sphere"""

    def __init__(self, num_groups: int, dim_params: int) -> None:
        """
        Initialize parameters spherical constraints

        Parameters
        ----------
        num_groups: int
            number of parameter groups
        dim_params: int
            dimension of each parameter group
        """
        num_eqs = 1
        super(SphericalConstraint, self).__init__(dim_params, num_eqs, num_groups)

    def __call__(self, params: Tensor) -> Tensor:
        """
        Zero map associated to spherical constraint

        Parameters
        ----------
        params: float-valued PyTorch tensor of shape [num_groups dim_params]
            parameters to be constrained to sphere

        Returns
        -------
        float-valued PyTorch tensor of shape [num_groups num_eqs]
            spherical constraint
        """
        return torch.sum(params**2, dim=-1).unsqueeze(1) - 1


class SCLinear(torch.nn.Module):

    """Example implementation constrained linear layer"""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        """
        Initialize constraint defining the manifold

        Parameters
        ----------
        dim_in: int
            input dimension linear layer
        dim_out: int
            output dimension linear layer
        bias: bool
            bias
        """
        super(SCLinear, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        # Initialize weights linear operator
        params = torch.rand(self.dim_out, self.dim_in)
        params = torch.einsum(
            "ij, i-> ij ", params, 1 / torch.sqrt(torch.sum(params**2, dim=-1))
        )
        self.params = ConstrainedParameter(
            init_params=params,
            constraint=SphericalConstraint(self.dim_out, self.dim_in),
        )

        # Bias
        if bias:
            self.bias = torch.nn.Parameter(torch.rand(self.dim_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluate linear layer

        Parameters
        ----------
        x: float-valued PyTorch tensor of shape [batch_size dim_in]
            input

        Returns
        -------
        float-valued PyTorch tensor of shape [batch_size dim_out]
            linear layer applied to input
        """
        return torch.nn.functional.linear(x, self.params, bias=self.bias)


class MLP(torch.nn.Module):

    """Constrained multilayer perceptron"""

    def __init__(
        self, dim_in: int, dim_latent: int, dim_out: int, num_hidden_layers: int
    ) -> None:
        """
        Initialize parameters

        Parameters
        ----------
        dim_in: int
            input dimension
        dim_latent: int
            latent dimension
        dim_out: int
            output dimension
        num_hidden_layers: int
            number of hidden layers
        """
        super(MLP, self).__init__()

        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.num_hidden_layers = num_hidden_layers

        # Input to latent dimension
        layers = torch.nn.ModuleList([])
        layers.append(SCLinear(self.dim_in, self.dim_latent))
        layers.append(torch.nn.LayerNorm(self.dim_latent))
        layers.append(torch.nn.GELU())

        # Hidden layers
        for layer_idx in range(self.num_hidden_layers):
            layers.append(SCLinear(self.dim_latent, self.dim_latent))
            layers.append(torch.nn.LayerNorm(self.dim_latent))
            layers.append(torch.nn.GELU())

        # Unconstrained output layer
        layers.append(torch.nn.Linear(self.dim_latent, self.dim_out))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
