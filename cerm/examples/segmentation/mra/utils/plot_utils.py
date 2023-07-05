"""Tools for plotting predictions"""

import torch
import matplotlib

from torch import Tensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
from torch.utils.tensorboard import SummaryWriter

from mra.network.model import ContourModel
from mra.utils import inference

# from wavelet_learner.utils.fourier_utils import FourierSeries
from cerm.examples.wavelets.dwt import (
    idwt1d,
    compute_order,
    high_pass_filter,
    determine_order_parity,
)

# Global plot settings

# Constants
fontsize = 12
color_approx = "teal"
color_detail = "steelblue"
color_contour_pred = "darkred"
color_contour_gt = "darkgreen"

# Initialize settings
matplotlib.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.size": fontsize}
)


def sample_wavelets(lpf: Tensor, num_levels: int) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    Sample mother and father wavelet on dyadic grid.

    Parameters
    ----------
    lpf: float-valued PyTorch Tensor of size [2m-1]
        low-pass filter
    num_levels: int
        number of levels upsampling
    """
    order = compute_order(lpf.shape[-1])
    lpf = lpf.unsqueeze(0).unsqueeze(0)
    hpf = high_pass_filter(lpf)

    # Approximation coefficients lowest resolution level
    # TODO: padding can be exactly determined by computing support of wavelet
    pad = (2 * order - 1, 2 * order - 1)
    init_mother = torch.nn.functional.pad(lpf, pad)
    init_father = torch.nn.functional.pad(hpf, pad)

    # Upsample
    mother_wavelet = idwt1d(
        init_mother.unsqueeze(0), [], lpf, num_levels_no_detail=num_levels
    )[-1].flatten()
    father_wavelet = idwt1d(
        init_father.unsqueeze(0), [], lpf, num_levels_no_detail=num_levels
    )[-1].flatten()

    # Associate grid
    res_level = num_levels + 1
    order_mother_wavelet = compute_order(mother_wavelet.shape[-1])
    grid_mother_wavelet = (
        torch.arange(1 - order_mother_wavelet, order_mother_wavelet) / 2**res_level
    )
    order_father_wavelet = compute_order(father_wavelet.shape[-1])
    grid_father_wavelet = (
        torch.arange(1 - order_father_wavelet, order_father_wavelet) / 2**res_level
    )

    return {
        "mother": (grid_mother_wavelet, 2 ** (res_level / 2) * mother_wavelet),
        "father": (grid_father_wavelet, 2 ** (res_level / 2) * father_wavelet),
    }


def plot_wavelets(
    lpf: Tensor,
    num_levels: int,
    figsize: Tuple[int, int] = (20, 8),
    set_title: bool = False,
    xlabel: str = "$t$",
    fontsize: int = 22,
):
    """
    Plot mother and father wavelet.

    Parameters
    ----------
    lpf: float-valued PyTorch Tensor of size [2m-1]
        low-pass filter
    num_levels: int
        number of levels upsampling
    figsize: Tuple[int, int]
        figure size
    set_title: bool
        if true set figure titles
    xlabel: str
        name horizontal axis
    """
    # Plot (need to crop this based on support wavelet)
    wavelets = sample_wavelets(lpf, num_levels)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(
        wavelets["mother"][0],
        wavelets["mother"][1],
        linewidth=2,
        color="steelblue",
    )
    ax[0].set_xlabel(xlabel, fontsize=fontsize)

    ax[1].plot(
        wavelets["father"][0],
        wavelets["father"][1],
        linewidth=2,
        color="darkgreen",
    )
    ax[1].set_xlabel(xlabel, fontsize=fontsize)

    for ax_idx in range(2):
        ax[ax_idx].tick_params(axis="x", labelsize=fontsize)
        ax[ax_idx].tick_params(axis="y", labelsize=fontsize)

    if set_title:
        ax[0].set_title(f"Father wavelet (scaling map)")
        ax[1].set_title(f"Mother wavelet")

    return fig, ax


def plot_refinement_masks(
    lpf: Tensor,
    period: float = 1.0,
    num_pts: int = 100,
    figsize: Tuple[int, int] = (20, 8),
    xlabel: str = "$\\xi$",
    fontsize: int = 22,
    colors: List[str] = ["steelblue", "darkgreen"],
) -> None:
    """
    Plot low and high pass filters H and G (1-periodic functions)

    Parameters
    ----------
    lpf: float-valued PyTorch Tensor of size [2m-1]
        low-pass filter
    period: float (constant)
        period of refinement masks
    num_pts: int
        number of discretization points [0, 1]
    """
    # Initialization
    hpf = high_pass_filter(lpf.unsqueeze(0).unsqueeze(0)).flatten()
    lpf = torch.nn.functional.pad(lpf, (1, 1))
    gridpts = torch.linspace(0, period, num_pts + 1)[
        0:num_pts
    ]  # no need to init this every time

    # Construct refinement masks H and G lpf and hpf
    filters = torch.flip(torch.stack((lpf, hpf)), dims=[-1])
    refinement_masks = FourierSeries(torch.stack((lpf, hpf)), period)(gridpts)
    magnitude = torch.abs(refinement_masks)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ylabels = ["$\\vert H(\\xi) \\vert$", "$\\vert G(\\xi) \\vert$"]
    for idx in range(magnitude.shape[0]):
        ax[idx].plot(gridpts, magnitude[idx], linewidth=2, color=colors[idx])
        ax[idx].set_xlabel(xlabel, fontsize=fontsize)
        ax[idx].set_ylabel(ylabels[idx], fontsize=fontsize)
        ax[idx].tick_params(axis="x", labelsize=fontsize)
        ax[idx].tick_params(axis="y", labelsize=fontsize)

    return fig, ax


def plot_contour(
    img: Tensor,
    contour_dyadic_scaled_gt: Tensor,
    contour_dyadic_scaled_pred: Tensor,
    init_res_level: int,
    figsize: Tuple[int, int],
    avg_midpoint_contour: Tensor = torch.zeros(2),
    linewidth_contour: int = 1.0,
    num_comp: int = 2,
    dpi: int = 100,
):
    """
    Plot groundtruth and predicted contour (wavelet decomposition)

    Parameters
    ----------
    img: float-valued PyTorch tensor of shape [heigth width]
        image containing roi
    contour_dyadic_scaled: Tensor
        (scaled) ground-truth contour
    approx_pred: List[Tensor]
        predicted approximation coefficients
    avg_midpoint_contour: float-valued PyTorch tensor of shape [2]
        average midpoint contours over training set
    init_res_level: int
        initial resolution level

    """
    plt.rc("axes", labelsize=fontsize)

    # Initialize canves for plotting contour
    height, width = img.shape
    figsize_img = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=figsize_img)
    ax.set_adjustable("datalim")
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    # Plot image
    ax.imshow(
        img,
        cmap="gray",
        aspect="equal",
        extent=(0, width, height, 0),
    )

    # Rescale approximation coefficients and contour
    avg_midpoint_contour = avg_midpoint_contour.view(2, 1)
    contour_gt = (
        2 ** (init_res_level / 2) * contour_dyadic_scaled_gt + avg_midpoint_contour
    )
    contour_pred = (
        2 ** (init_res_level / 2) * contour_dyadic_scaled_pred + avg_midpoint_contour
    )

    # Plot contours
    ax.plot(
        contour_pred[1],
        contour_pred[0],
        linewidth=linewidth_contour,
        color=color_contour_pred,
    )
    ax.plot(
        contour_gt[1],
        contour_gt[0],
        linewidth=linewidth_contour,
        color=color_contour_gt,
    )

    return fig, ax


def plot_dwt_contour(
    contour_dyadic_scaled_gt: Tensor,
    approx_pred: List[Tensor],
    detail_pred: List[Tensor],
    init_res_level: int,
    figsize: Tuple[int, int],
    linewidth: int = 2.0,
    num_comp: int = 2,
    dpi: int = 150,
):
    """
    Plot wavelet decomposition of predicted contour

    Parameters
    ----------
    contour_dyadic_scaled: Tensor
        (scaled) ground-truth contour
    approx_pred: List[Tensor]
        predicted approximation coefficients
    detail_pred: List[Tensor]
        predicted detail coefficients
    """
    num_levels_detail = len(detail_pred)
    num_levels_approx = len(approx_pred)

    fig, ax = plt.subplots(num_levels_approx, 4, figsize=figsize)

    # Export coefficients
    for comp_idx in range(num_comp):
        fig_approx_col_idx = 2 * comp_idx
        fig_detail_col_idx = fig_approx_col_idx + 1

        # Plot predicted approximation coefficients at all levels
        for level in range(num_levels_approx):
            ax[level, fig_approx_col_idx].plot(
                approx_pred[num_levels_approx - level - 1][comp_idx],
                linewidth=linewidth,
                color=color_contour_pred,
            )
            ax[level, fig_approx_col_idx].set_title(
                f"Approx level {init_res_level - level}"
            )

        # Plot ground-truth approximation coefficients (only available at highest level)
        ax[0, fig_approx_col_idx].plot(
            contour_dyadic_scaled_gt[comp_idx],
            linewidth=linewidth,
            color=color_contour_gt,
        )

        ax[0, fig_approx_col_idx].legend(["Groundtruth", "Prediction"])

        # Predicted detail coefficients
        for level in range(num_levels_detail):
            res_level = num_levels_approx - num_levels_detail + level
            ax[res_level, fig_detail_col_idx].plot(
                detail_pred[num_levels_detail - level - 1][comp_idx],
                linewidth=linewidth,
                color=color_detail,
            )
            ax[res_level, fig_detail_col_idx].set_title(
                f"Detail level {init_res_level - res_level}"
            )

    return fig, ax


def plot_contours_tensorboard(
    writer: SummaryWriter,
    model: ContourModel,
    dataloader: DataLoader,
    avg_midpoint_contour: Tensor,
    data_type: str,
    epoch: int,
    device: str,
    figsize: Tuple[int, int] = (9, 9),
) -> None:
    """
    Plot contours amd wavelet decompositions on tensorboard

    Parameters
    ----------
    writer: SummaryWriter
        writer to tensorboard
    model: ContourModel
        contour model
    dataloader: DataLoader
        dataloader for samples to monitor
    avg_midpoint_contour: float-valued PyTorch tensor of shape [2]
        average midpoint contours over training set
    data_type: str in {"train", "val"}
        indicates whether dataloader provided train or validation samples
    epoch: int
        current epoch
    """
    with torch.no_grad():
        obs, pred = inference.predict(model, dataloader, device)

        for obs_batch, pred_batch in zip(obs, pred):
            batch_size = obs_batch["img"].shape[0]

            for batch_idx in range(batch_size):
                sample_idx = obs_batch["idx"][batch_idx]

                fig, ax = plot_contour(
                    obs_batch["img"][batch_idx][0],
                    obs_batch["contour_dyadic_scaled"][batch_idx],
                    pred_batch["approx"][-1][batch_idx],
                    model.init_res_level,
                    figsize,
                    avg_midpoint_contour=avg_midpoint_contour,
                )

                writer.add_figure(
                    f"contour/{data_type}/sample_idx_{sample_idx}",
                    fig,
                    global_step=epoch,
                )

                plt.close(fig)

                fig, ax = plot_dwt_contour(
                    obs_batch["contour_dyadic_scaled"][batch_idx],
                    [a[batch_idx] for a in pred_batch["approx"]],
                    [d[batch_idx] for d in pred_batch["detail"]],
                    model.init_res_level,
                    figsize,
                )

                writer.add_figure(
                    f"dwt/{data_type}/sample_idx_{sample_idx}",
                    fig,
                    global_step=epoch,
                )


def plot_wavelets_tensorboard(
    writer: SummaryWriter,
    model: ContourModel,
    epoch: int,
    figsize: Tuple[int, int] = (9, 9),
    num_levels: int = 5,
) -> None:
    """
    Plot learned wavelets on tensorboard

    Parameters
    ----------
    writer: SummaryWriter
        writer to tensorboard
    model: ContourModel
        contour model
    epoch: int
        current epoch
    figsize: Tuple[int, int]
        firgure size
    num_levels: int
        number of levels used in upsampling to display wavelets
    """
    # Group low pass filters by spatial dimension and remove num_filter dimension
    lpf = model.wavelet_layer.group_lpfs().squeeze(1)

    # Plot per spatial component (channel dimension)
    for comp_idx in range(lpf.shape[0]):
        lpf_comp = lpf[comp_idx].detach().cpu()

        # Plot wavelets
        fig, ax = plot_wavelets(lpf_comp, num_levels)
        writer.add_figure(
            f"wavelets/spatial_comp_{comp_idx}",
            fig,
            global_step=epoch,
        )
        plt.close(fig)

        # Plot refinement masks
        # fig, ax = plot_refinement_masks(lpf_comp)
        # writer.add_figure(
        #     f"refinement_mask/spatial_comp_{comp_idx}",
        #     fig,
        #     global_step=epoch,
        # )
        # plt.close(fig)
