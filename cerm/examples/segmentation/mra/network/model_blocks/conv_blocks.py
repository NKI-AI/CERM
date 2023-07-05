"""Convolutional building blocks"""

import torch
import numpy as np

from torch import Tensor
from typing import Tuple, Callable, List, Union

from mra.network.model_blocks.init_utils import init_activation, init_conv_weights


class ResBlock(torch.nn.Module):
    """Residual convolutional block."""

    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        group_norm: bool = False,
        num_groups: int = 1,
        dropout_prob: float = 0,
        activation: str = "ReLU",
        dilation: int = 1,
        dim_kernel: int = 2,
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
        group_norm: bool, optional
            whether to use group normalization or not
        num_groups: int
            number of groups in which channels will be divided
        dropout_prob: float
            dropout probability
        activation: str
            activation function used after each layer
        dilation: int
            dilation convolutional kernel
        dim_kernel: int
            dimension of kernel
        """
        super(ResBlock, self).__init__()

        # Dimensions
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_layers = num_layers
        self.dim_kernel = dim_kernel

        # Kernel
        self.conv = torch.nn.Conv3d if dim_kernel == 3 else torch.nn.Conv2d
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (self.kernel_size - 1) // 2

        # Activation map and regularization
        self.activation = init_activation(activation)
        self.group_norm = group_norm
        self.num_groups = num_groups
        if self.group_norm:
            self.bias = False
        else:
            self.bias = True
        self.dropout_prob = dropout_prob

        # Convolution layers
        layers = torch.nn.ModuleList([])
        in_channels = self.num_in_channels

        for layer_idx in range(num_layers):
            layers.append(
                self.conv(
                    in_channels,
                    self.num_out_channels,
                    kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                    dilation=self.dilation,
                )
            )

            layers[-1].apply(init_conv_weights)

            if self.group_norm:
                layers.append(
                    torch.nn.GroupNorm(self.num_groups, self.num_out_channels)
                )

            # Note: dropout has a different effect on convolutions
            if self.dropout_prob > 0:
                layers.append(torch.nn.Dropout(p=self.dropout_prob))

            if layer_idx < num_layers - 1:
                layers.append(self.activation)

            if layer_idx == 0:
                in_channels = self.num_out_channels

        self.layers = torch.nn.Sequential(*layers)

        # If number channels input and number kernels differ use 1x1 conv to align.
        if self.num_in_channels == self.num_out_channels:
            self.skip_layer = torch.nn.Identity()
        else:
            self.skip_layer = self.conv(
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


class Encoder(torch.nn.Module):
    """Convolutional encoder"""

    def __init__(
        self,
        input_dim: Union[Tuple[int, int], Tuple[int, int, int]],
        len_encoder: int,
        len_res_block: int,
        init_num_kernels: int,
        group_norm: bool = False,
        num_groups: int = 4,
        activation: str = "ReLU",
        dropout_prob: float = 0,
        kernel_size: int = 3,
        len_res_layer: int = 2,
    ) -> None:
        """
        Initialize encoder.

        Parameters
        ----------
        input_dim: Union(Tuple[int, int], Tuple[int, int, int])
            spatial dimensions of input image ([depth height width] or [height width])
        len_encoder: int
            number of down-sampling blocks
        len_res_block: int
            number of residual conv-blocks in a down-sampling block
        init_num_kernels: int
            initial number of kernels
        group_norm: bool, optional
            specifies whether to use group normalization or not
        num_groups: int
            number of groups in which channels are divided
        activation: str, optional
            activation function used after each layer
        dropout: float, optional
            dropout probability in fully connected layers
        kernel_size: int, optional
            filter size
        len_res_layer: int, optional
            number of conv layers in a residual block
        """
        super(Encoder, self).__init__()

        # Architecture
        self.input_dim = input_dim
        self.len_encoder = len_encoder
        self.len_res_block = len_res_block
        self.len_res_layer = len_res_layer
        self.init_num_kernels = init_num_kernels
        self.kernel_size = kernel_size
        self.spatial_dim = len(self.input_dim)

        # Regularization
        self.activation = activation
        self.group_norm = group_norm
        self.num_groups = num_groups
        self.dropout_prob = dropout_prob

        # Downsampling
        self.dim_out_encoder = []
        self.pool_kernel_size = tuple((2 for _ in range(self.spatial_dim)))
        self.pool_stride = tuple((2 for _ in range(self.spatial_dim)))
        self.conv = torch.nn.Conv3d if self.spatial_dim == 3 else torch.nn.Conv2d

        # Construct encoder
        self.encoder = torch.nn.ModuleList([])
        self.downsample = torch.nn.ModuleList([])
        dim_in_block = self.input_dim
        num_out_channels = self.init_num_kernels
        num_in_channels = 1
        for block_idx in range(self.len_encoder):
            blocks = torch.nn.ModuleList([])

            # Residual block
            for rep_idx in range(self.len_res_block):
                blocks.append(
                    ResBlock(
                        num_in_channels,
                        num_out_channels,
                        group_norm=self.group_norm,
                        num_groups=self.num_groups,
                        dim_kernel=self.spatial_dim,
                        dropout_prob=self.dropout_prob,
                    )
                )
                if rep_idx == 0:
                    num_in_channels = num_out_channels

            self.encoder.append(torch.nn.Sequential(*blocks))

            self.downsample.append(
                self.conv(
                    num_out_channels,
                    num_out_channels,
                    self.kernel_size,
                    stride=2,
                    padding=1,
                )
            )

            # Store output dimensions
            self.dim_out_encoder.append([num_out_channels, *dim_in_block])
            dim_in_block = [
                dim_in_block[dim_idx] // 2 for dim_idx in range(self.spatial_dim)
            ]
            self.dim_bottom = (num_out_channels, *dim_in_block)

            if block_idx % 2 == 1:
                num_out_channels *= 2

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        input = x
        for downsample, conv_layer in zip(self.downsample, self.encoder):
            out = conv_layer(input)
            skips.append(out)
            input = downsample(out)
        return input, skips
