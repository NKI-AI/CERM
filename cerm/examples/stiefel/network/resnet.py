"""Convolutional building blocks"""

import torch
import numpy as np

from torch import Tensor
from typing import Tuple, Callable, List, Union

from cerm.examples.stiefel.network.stiefel import StiefelConv2d


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


class ResBlock(torch.nn.Module):
    """Residual convolutional block."""

    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "GELU",
        batch_norm: bool = True,
        dilation: int = 1,
        stiefel: bool = False,
    ) -> None:
        """
        Initialize dimensions of residual convolution block.

        Parameters
        ----------
        num_in_channels: int
            number of input channels
        num_out_channels: int
            number of output channels
        num_layers: int
            number of convolutional layers in block
        kernel_size: int
            size of kernel
        stride: int
            stride used in cross-correlation with kernel
        activation: str
            activation function used after each layer
        dilation: int
            dilation convolutional kernel
        stiefel: bool
            use stiefel constraint is true
        """
        super(ResBlock, self).__init__()

        # Dimensions
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_layers = num_layers
        self.stiefel = stiefel

        # Kernel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (self.kernel_size - 1) // 2

        # Activation map and regularization
        self.batch_norm = batch_norm
        self.activation = init_activation(activation)
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True

        self.conv2d = StiefelConv2d if self.stiefel else torch.nn.Conv2d

        # Convolution layers
        layers = torch.nn.ModuleList([])
        in_channels = self.num_in_channels

        for layer_idx in range(num_layers):
            layers.append(torch.nn.BatchNorm2d(in_channels))

            layers.append(self.activation)

            layers.append(
                self.conv2d(
                    in_channels,
                    self.num_out_channels,
                    kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                )
            )

            if layer_idx == 0:
                in_channels = self.num_out_channels

        self.layers = torch.nn.Sequential(*layers)

        # If number channels input and number kernels differ use 1x1 conv to align.
        if self.num_in_channels == self.num_out_channels:
            self.skip_layer = torch.nn.Identity()
        else:
            self.skip_layer = torch.nn.Conv2d(
                self.num_in_channels,
                self.num_out_channels,
                1,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate residual convolutional block."""
        return self.activation(self.skip_layer(x) + self.layers(x))


class WideResNet(torch.nn.Module):
    """Convolutional encoder."""

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        widen_factor: int,
        num_res_blocks: int,
        num_classes: int,
        batch_norm: bool = True,
        activation: str = "GELU",
        kernel_size: int = 3,
        len_res_layer: int = 2,
        num_kernels: Tuple[int, int, int] = (16, 32, 64),
        stiefel: bool = False,
    ) -> None:
        """
        Initialize WideResnet.

        Parameters
        ----------
        input_dim: Tuple[int, int, int]
            dimensions input image ([num_channels height width])
        widen_factor: int
            multiplication factor number kernels
        num_res_blocks: int
            number of residual layers in a block
        activation: str, optional
            activation function used after each layer
        kernel_size: int, optional
            filter size
        len_res_layer: int, optional
            number of conv layers in a residual block
        num_kernels: List[int, int ,int]
            number of kernels per layer
        stiefel: bool
            use stiefel constraint is true
        """
        super(WideResNet, self).__init__()

        # Architecture
        self.input_dim = input_dim
        self.widen_factor = widen_factor
        self.kernel_size = kernel_size
        self.num_kernels = [self.widen_factor * n for n in num_kernels]
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes

        # Regularization
        self.activation = activation
        self.batch_norm = batch_norm
        self.stiefel = stiefel

        # Construct network
        self.dim_out = int(
            np.prod(np.array(input_dim[1:]) / 2 ** (len(self.num_kernels)))
            * self.num_kernels[-1]
        )

        # First resblock; no repetitions
        num_in_channels = self.input_dim[0]
        blocks = torch.nn.ModuleList([])
        blocks.append(
            ResBlock(
                num_in_channels,
                self.num_kernels[0],
                kernel_size=self.kernel_size,
                activation=self.activation,
                batch_norm=self.batch_norm,
                stiefel=self.stiefel,
            )
        )

        # Resblock with repetitions and down sampling
        num_in_channels = self.num_kernels[0]
        for block_idx, num_features in enumerate(self.num_kernels):
            for rep_idx in range(self.num_res_blocks):
                blocks.append(
                    ResBlock(
                        num_in_channels,
                        num_features,
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        batch_norm=self.batch_norm,
                        stiefel=self.stiefel,
                    )
                )
                if rep_idx == 0:
                    num_in_channels = num_features
            if block_idx < len(self.num_kernels) - 1:
                blocks.append(torch.nn.AvgPool2d(2))
            num_in_channels = num_features

        blocks.append(torch.nn.AvgPool2d(8))
        self.convnet = torch.nn.Sequential(*blocks)
        self.classifier = torch.nn.Linear(self.num_kernels[-1], self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.convnet(x).flatten(start_dim=1))
