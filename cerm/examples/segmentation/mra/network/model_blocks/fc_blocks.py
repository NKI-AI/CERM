"""Fully connectd building blocks."""

import torch
import numpy as np

from torch import Tensor
from typing import Tuple, Callable, List, Union

from mra.network.model_blocks.init_utils import init_activation


class MLP(torch.nn.Module):
    """Fully connected (residual) MLP for final predictions."""

    def __init__(
        self,
        dim_in: Union[Tuple[int, int, int], int],
        dim_latent: int,
        dim_out: Union[Tuple[int, int, int], int],
        activation: str = "ReLU",
        num_hidden_layers: int = 3,
        num_channels_compress: int = 0,
        reduce_latent_dim: bool = False,
    ) -> None:
        """
        Initialize dimensions prediction branch.

        Parameters
        ----------
        dim_in: Union[Tuple[int, int, int], int]
            shape of input (3d if input is a 2d feature map)
        dim_latent: int
            number of hidden components
        dim_out: dim_out: Union[Tuple[int, int, int], int]
            shape of output (3d if desired output is a 2d feature map)
        num_hidden_layers: int
            number of hidden layers
        num_channels_compress:
            number of channels to compress if input is 3d
        """
        super(MLP, self).__init__()

        # Dimensions
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        if type(self.dim_out) == int:
            self.dim_out = (self.dim_out,)
        self.dim_out_flat = np.prod(dim_out)

        # Architecture
        self.num_hidden_layers = num_hidden_layers
        self.num_channels_compress = num_channels_compress
        self.activation = init_activation(activation)
        self.reduce_latent_dim = reduce_latent_dim

        # Compress channels to reduce memory footprint
        if self.num_channels_compress > 0 and type(self.dim_in) == int:
            raise ValueError("There are no channels to compress for 1d input")

        if self.num_channels_compress > 0:
            self.compress = torch.nn.Conv2d(
                self.dim_in[0],
                self.num_channels_compress,
                1,
                stride=1,
                padding=0,
            )
            self.dim_in_flat = self.num_channels_compress * np.prod(self.dim_in[1:])
        else:
            self.compress = torch.nn.Identity()
            self.dim_in_flat = np.prod(self.dim_in)

        # Layer mapping input to latent representation
        self.to_latent = torch.nn.Sequential(
            self.compress,
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(self.dim_in_flat, self.dim_latent),
        )

        # Layers mapping hidden representation to desired output
        dim_latent = self.dim_latent
        self.fc_layers = torch.nn.ModuleList([])
        for _ in range(num_hidden_layers):
            if self.reduce_latent_dim:
                dim_latent_next = dim_latent // 2
            else:
                dim_latent_next = dim_latent
            self.fc_layers.append(torch.nn.Linear(dim_latent, dim_latent_next))
            dim_latent = dim_latent_next
        self.fc_layers.append(torch.nn.Linear(dim_latent, self.dim_out_flat))

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        y = self.activation(self.to_latent(x))
        for fc_layer in self.fc_layers[0:-1]:
            y = self.activation(fc_layer(y) + y)
        y = self.fc_layers[-1](y).view(bsize, *self.dim_out)
        return y
