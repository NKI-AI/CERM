"""General tools setting up network."""

import torch

from typing import Callable, Union


def init_activation(activation: str, inplace: bool = True) -> Callable:
    """
    Initialize activation function.

    Parameters
    ----------
    activation: str
        name of activation function

    Returns
    -------
    PyTorch Module
        activation function
    """
    if activation == "ReLU":
        return torch.nn.ReLU(inplace=inplace)
    elif activation == "Tanh":
        return torch.nn.Tanh(inplace=inplace)
    elif activation == "GELU":
        return torch.nn.GELU()
    elif activation == "PReLU":
        return torch.nn.PReLU()
    else:
        raise NotImplementedError("Prescribed activation is not yet implemented")


def init_conv_weights(conv_layer: torch.nn.Conv2d) -> None:
    """Xavier initialization."""
    if isinstance(conv_layer, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(conv_layer.weight.data)
    else:
        raise NotImplementedError("Xavier normalization only implemented for conv2d")
