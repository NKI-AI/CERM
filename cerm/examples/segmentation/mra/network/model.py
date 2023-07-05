"""Implementation of contour network."""

import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Tuple, List, Union, Dict
from torch.nn import functional as F

from mra.network.model_blocks.init_utils import init_activation
from mra.network.model_blocks import conv_blocks, fc_blocks
from cerm.examples.wavelets.wavelet_layer import WaveletLayer1d


class ContourModel(torch.nn.Module):
    """Neural network predicting wavelet decompositions of contours"""

    def __init__(
        self,
        dim_in: Tuple[int, int],
        order_wavelet: int,
        init_res_level: int,
        num_levels_down: int,
        num_levels_up: int,
        len_encoder: int,
        len_res_block: int,
        init_num_kernels: int,
        num_channels_compress: int = 32,
        num_wavelet_filters: int = 1,
        mlp_contour_depth: int = 2,
        mlp_contour_dim_latent: int = 64,
        mlp_classifier_depth: int = 2,
        mlp_classifier_dim_latent: int = 64,
        group_norm: bool = False,
        num_groups: int = 4,
        activation: str = "ReLU",
        dropout_prob: int = 0,
        threshold_detail: float = 1e-02,
        device: str = "cpu",
    ) -> None:
        """
        Initialize parameters contour model

        Parameters
        ----------
        dim_in: Tuple[int, int] (heigth, width)
            size of input
        order_wavelet: int
            order of wavelet
        init_res_level: int
            initial (highest) resolution level signal
        num_levels: int
            number of levels in upsampling path
        len_encoder: int
            number of down-sampling blocks
        len_res_block: int
            number of residual conv-blocks in a down-sampling block.
        init_num_kernels: int
            initial number of kernels used in first convolutional block
        num_channels_compress: optional
            number of channels output skip-connection is compressed to
        mlp_contour_depth: int, optional
            length (number of layers) of the final prediction branch contour
        mlp_contour_dim_latent: int, optional
            dimension hidden layers MLP contour branch
        mlp_classifier_depth: int, optional
            length (number of layers) of the final prediction branch classifier
        mlp_classifier_dim_latent: int, optional
            dimension hidden layers MLP classification branch
        group_norm: bool, optional
            specifies whether to use group normalization or not in convolutional layers
        num_groups: int, optional
            number of groups in which channels are divided
        activation: str, optional
            name activation function used in layers
        dropout_prob: float, optional
            dropout probability
        threshold_detail:
            discard too small detail coefficients
        device: str, optional
            device on which computations are stored
        """
        super(ContourModel, self).__init__()

        # Dimensions
        self.dim_in = dim_in
        self.init_res_level = init_res_level
        self.init_res_level_with_detail = (
            self.init_res_level - num_levels_down + num_levels_up
        )
        self.num_levels_down = num_levels_down
        self.num_levels_up = num_levels_up
        self.num_levels_no_detail = num_levels_down - num_levels_up
        self.num_comp_contour = 2

        # Dimensions convolutional blocks
        self.len_encoder = len_encoder
        self.len_res_block = len_res_block
        self.mlp_contour_depth = mlp_contour_depth
        self.mlp_contour_dim_latent = mlp_contour_dim_latent
        self.mlp_classifier_depth = mlp_classifier_depth
        self.mlp_classifier_dim_latent = mlp_classifier_dim_latent
        self.init_num_kernels = init_num_kernels
        self.num_channels_compress = num_channels_compress
        self.init_pred_level = len_encoder - num_levels_up + 1

        # Regularization
        self.activation = init_activation(activation)
        self.group_norm = group_norm
        self.num_groups = num_groups
        self.dropout_prob = dropout_prob
        self.threshold_detail = threshold_detail

        # Construct base network: encoder and bottleneck
        self.encoder = conv_blocks.Encoder(
            self.dim_in,
            self.len_encoder,
            self.len_res_block,
            self.init_num_kernels,
            group_norm=self.group_norm,
            num_groups=self.num_groups,
            activation=activation,
        )

        # Attach branches
        self._construct_classifier_branch()
        self._construct_detail_branch()
        self._construct_approx_branch()

        # Wavelet layer for reconstruction
        num_filters_per_channel = 1
        self.wavelet_layer = WaveletLayer1d(
            order_wavelet,
            num_levels_down,
            num_filters_per_channel,
            periodic_signal=True,
            num_channels=2,
        )

    def _construct_classifier_branch(self) -> None:
        """Construct classifier."""
        dim_out = 1
        self.classifier = fc_blocks.MLP(
            self.encoder.dim_bottom,
            self.mlp_classifier_dim_latent,
            dim_out,
            num_hidden_layers=self.mlp_classifier_depth,
            num_channels_compress=self.num_channels_compress,
        )

    def _construct_detail_branch(self) -> None:
        """Detail coefficients from skip layers."""
        self.detail = torch.nn.ModuleList([])

        for block_idx in range(self.init_pred_level, self.len_encoder + 1):
            # Input dimension from skip or bottom
            if block_idx < self.len_encoder:
                dim_in_pred = self.encoder.dim_out_encoder[block_idx]
                num_channels_compress = self.num_channels_compress
            else:
                dim_in_pred = self.encoder.dim_bottom
                num_channels_compress = 0

            # Number of wavelet coefficients to predict
            dim_out = 2 ** (
                self.init_res_level_with_detail + self.init_pred_level - block_idx - 1
            )

            detail = torch.nn.ModuleList([])
            for spatial_comp in range(self.num_comp_contour):
                detail.append(
                    fc_blocks.MLP(
                        dim_in_pred,
                        self.mlp_contour_dim_latent,
                        int(dim_out),
                        num_hidden_layers=self.mlp_contour_depth,
                        num_channels_compress=num_channels_compress,
                    )
                )

            self.detail.append(detail)

    def _construct_approx_branch(self) -> None:
        """Construct approximation layer."""
        self.approx = torch.nn.ModuleList([])
        dim_out = int(2 ** (self.init_res_level - self.num_levels_down))
        for spatial_comp in range(self.num_comp_contour):
            self.approx.append(
                fc_blocks.MLP(
                    self.encoder.dim_bottom,
                    self.mlp_contour_dim_latent,
                    dim_out,
                    num_hidden_layers=self.mlp_contour_depth,
                )
            )

    def compute_detail(self, block_idx: int, out_skip: Tensor) -> Tensor:
        """Compute detail coefficients."""
        return torch.stack(
            [
                self.detail[block_idx][spat_comp](out_skip)
                for spat_comp in range(self.num_comp_contour)
            ],
            dim=1,
        ).unsqueeze(2)

    def compute_approx(self, out_encoder: Tensor) -> Tensor:
        """Compute approximation coefficients lowest level."""
        return torch.stack(
            [
                self.approx[spat_comp](out_encoder)
                for spat_comp in range(self.num_comp_contour)
            ],
            dim=1,
        ).unsqueeze(2)

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """Reconstruct surface using features extracted from encoder"""
        device = x.device
        batch_size = x.size()[0]
        input = x

        # Output encoder
        out_encoder, skips = self.encoder(x)
        preprob = self.classifier(out_encoder)

        # Downsampling path encoder: compute detail coefficients from skip layers
        detail = []
        for block_idx in range(self.init_pred_level, self.len_encoder):
            pred_level = block_idx - self.init_pred_level
            detail.insert(
                0,
                self.compute_detail(pred_level, skips[block_idx]),
            )

        # Detail coefficients lowest level
        if self.init_pred_level == self.len_encoder:
            pred_level = 0
        else:
            pred_level += 1
        detail.insert(
            0,
            self.compute_detail(pred_level, out_encoder),
        )

        # Approximation coefficients higher levels
        init_approx = self.compute_approx(out_encoder)
        approx = self.wavelet_layer.idwt(
            init_approx, detail, num_levels_no_detail=self.num_levels_no_detail
        )

        # Get rid of redundant num filter dimension
        approx = [a.squeeze(2) for a in approx]
        detail = [d.squeeze(2) for d in detail]

        # Threshold detail coefficients
        for d in detail:
            d[torch.abs(d) < self.threshold_detail] *= 0.0

        return {
            "classifier": preprob,
            "approx": approx,
            "detail": detail,
        }
