"""Implementation of wavelet layers"""

import torch

from typing import Tuple, Dict, List
from torch import Tensor

from cerm.network.constrained_params import ConstrainedParameter
from cerm.examples.wavelets import dwt
from cerm.examples.wavelets.qmf_constraints import QMFConstraint


class WaveletBaseLayer(torch.nn.Module):
    """Base class for n-dimensional wavelet layer"""

    def __init__(
        self,
        dim: int,
        order: int,
        num_levels_down: int,
        num_filters_per_channel: int,
        num_channels: int = 1,
        periodic_signal: bool = False,
    ) -> None:
        """
        Initialize filters wavelet layer

        Parameters
        ----------
        dim: int
            dimension of wavelet decomposition
        order: int
            order of filters
        num_filters_per_channel: int
            number of wavelet filters
        number_of_decomps: int
            number of levels in wavelet decompositions
        num_channels: int
            number of channels in input signal
        periodic_signal: bool
            indicates whether the input signal is periodic
        """
        super(WaveletBaseLayer, self).__init__()

        # Dimensions
        self.dim = dim
        self.order = order

        # Parameters dwt
        self.num_levels_down = num_levels_down
        self.num_filters_per_channel = num_filters_per_channel
        self.num_channels = num_channels
        self.periodic_signal = periodic_signal

        # Initialize constraint
        num_filters = self.dim * num_filters_per_channel * num_channels
        self.dim_filter = 2 * self.order - 1
        self.lpf = ConstrainedParameter(
            constraint=QMFConstraint(num_filters, self.order)
        )


class WaveletLayer1d(WaveletBaseLayer):
    """One-dimensional wavelet layer"""

    def __init__(
        self,
        order: int,
        num_levels_down: int,
        num_filters_per_channel: int,
        num_channels: int = 1,
        periodic_signal: bool = False,
    ) -> None:
        """
        Initialize filters 1d wavelet layer

        Parameters
        ----------
        order: int
            order of filter
        num_levels_down: int
            number of levels in wavelet decompositions
        num_filters_per_channel: int
            number of wavelet filters
        num_channels: int
            number of channels in input signal
        periodic_signal: bool
            indicates whether the input signal is periodic
        """
        super(WaveletLayer1d, self).__init__(
            1,
            order,
            num_levels_down,
            num_filters_per_channel,
            num_channels=num_channels,
            periodic_signal=periodic_signal,
        )

    def group_lpfs(self) -> Tensor:
        """
        Reshape stacked low pass filters into shape [num_channels num_filters dim_filter]

        Returns
        -------
        float-valued PyTorch tensor of shape [num_channels num_filters dim_filter]
            low pass filters grouped by channel and number of filters
        """
        return self.lpf.view(
            self.num_channels,
            self.num_filters_per_channel,
            self.dim_filter,
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Compute wavelet decomposition of 1d signal

        Parameters
        ----------
        x: float-valued PyTorch Tensor of size [batch_dims num_channels 2^k]

        Returns
        -------
        approx: List[Tensor]
            approximation coefficients
        detail: List[Tensor]
            detail coefficients
        """
        return dwt.dwt1d(
            x,
            self.group_lpfs(),
            self.num_levels_down,
            periodic_signal=self.periodic_signal,
        )

    def idwt(
        self,
        init_approx: Tensor,
        detail: List[Tensor],
        parity_init_order: bool = None,
        num_levels_no_detail: int = 0,
    ) -> Dict[str, Tensor]:
        """
        Wrapper inverse discrete wavelet transform

        Parameters
        ----------
        approx: float-valued PyTorch Tensor of size [batch_size num_channels num_filters m]
            approximation coefficients at lowest resolution level
        detail: List[Tensor]
            detail coefficients
        parity_init_order: bool
            parity of order signal initial level (only needed in non-periodic case)

        Returns
        -------
        approx: List[Tensor]
            reconstruction approximation coefficients at subsequent resolution levels
        """
        return dwt.idwt1d(
            init_approx,
            detail,
            self.group_lpfs(),
            parity_init_order=parity_init_order,
            periodic_signal=self.periodic_signal,
            num_levels_no_detail=num_levels_no_detail,
        )


class WaveletLayerSeparable2d(WaveletBaseLayer):
    """Two-dimensional separable wavelet layer"""

    def __init__(
        self,
        order: int,
        num_levels_down: int,
        num_filters_per_channel: int,
        num_channels: int = 1,
        periodic_signal: bool = False,
    ) -> None:
        """
        Initialize filters 2d separable wavelet layer

        Parameters
        ----------
        order: int
            order of filter
        num_filters_per_channel: int
            number of wavelet filters
        number_of_decomps: int
            number of levels in wavelet decompositions
        num_channels: int
            number of channels in input image
        periodic_signal: bool
            indicates whether the input signal is periodic
        """
        super(WaveletLayerSeparable2d, self).__init__(
            2,
            order,
            num_levels_down,
            num_filters_per_channel,
            num_channels=num_channels,
            periodic_signal=periodic_signal,
        )

    def group_lpfs(self) -> Tuple[Tensor, Tensor]:
        """
        Separate stacked low pass filters into channel and spatial dimensions

        Returns
        -------
        List[Tensor, Tensor]
            low pass filters separated into channel and spatial dimensions
        """
        return [
            self.lpf[: self.num_filters_per_channel * self.num_channels].view(
                self.num_channels,
                self.num_filters_per_channel,
                self.dim_filter,
            ),
            self.lpf[self.num_filters_per_channel * self.num_channels :].view(
                self.num_channels,
                self.num_filters_per_channel,
                self.dim_filter,
            ),
        ]

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Dict[str, List[Tensor]]]:
        """
        Compute wavelet decomposition of two-dimensional input tensor

        Parameters
        ----------
        x: float-valued PyTorch Tensor of size [batch_dims num_channels h w]
            two-dimensional input signal

        Returns
        -------
        approx: List[Tensor]
            approximation coefficients
        detail: List[Dict[str, Tensor]]
            detail coefficients
        """
        return dwt.dwt2d(
            x,
            self.group_lpfs(),
            num_levels=self.num_levels_down,
            periodic_signal=self.periodic_signal,
        )

    def idwt(
        self,
        init_approx: Tensor,
        detail: Dict[str, Tensor],
        parity_init_order: Tuple[bool, bool] = (1, 1),
        num_levels_no_detail: int = 0,
    ) -> Dict[str, Tensor]:
        """
        Wrapper around inverse dwt

        Parameters
        ----------
        approx: float-valued PyTorch Tensor of size [batch_size num_channels num_filters 2m-1 2m-1]
            approximation coefficients at lowest resolution level
        detail: Dict[str, Tensor]
            detail coefficients
        parity_init_order: Tuple[bool, bool]
            parity of order signal initial level (only needed in non-periodic case)
        num_level_no_detail: int
            number of resolution levels up after exhausting provided detail coefficients

        Returns
        -------
        approx: List[Tensor]
            reconstruction / upsampling approximation at subsequent resolution levels
        """
        return dwt.idwt2d(
            init_approx,
            detail,
            self.group_lpfs(),
            parity_init_order=parity_init_order,
            periodic_signal=self.periodic_signal,
            num_levels_no_detail=num_levels_no_detail,
        )
