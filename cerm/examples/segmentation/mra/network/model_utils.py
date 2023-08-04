"""Model utilities"""

import logging
import torch
import numpy as np

from pathlib import Path
from omegaconf import DictConfig

from mra.network.model import ContourModel


def init_model(cfg: DictConfig) -> ContourModel:
    """
    Initialize neural netwerk.

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict

    Returns
    -------
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contours
    """
    # Contour layer
    return ContourModel(
        cfg.dataset.dim_img,
        cfg.network.decoder.order_wavelet,
        cfg.network.decoder.init_res_level,
        cfg.network.decoder.num_levels_down,
        cfg.network.decoder.num_levels_up,
        cfg.network.encoder.len_encoder,
        cfg.network.encoder.len_res_block,
        cfg.network.encoder.init_num_kernels,
        num_channels_compress=cfg.network.decoder.num_channels_compress,
        mlp_contour_depth=cfg.network.mlp_contour.depth,
        mlp_contour_dim_latent=cfg.network.mlp_contour.dim_latent,
        group_norm=cfg.network.regularization.group_norm,
        num_groups=cfg.network.regularization.num_groups,
        activation=cfg.network.activation,
        dropout_prob=cfg.network.regularization.dropout_prob,
        threshold_detail=cfg.prep.threshold_detail,
        device=cfg.setup.device,
    ).to(cfg.setup.device)


def load_checkpoint(cfg: DictConfig, model: ContourModel, epoch: int = None) -> None:
    """
    Load model checkpoint.

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contours
    epoch: int
        epoch to be loaded
    """
    # Overide epoch in config if another one is supplied
    if type(epoch) == int:
        model_epoch = epoch
    else:
        model_epoch = cfg.setup.epoch

    base_model_dir = (
        Path(cfg.setup.model_dir)
        / "checkpoints"
        / f"model_checkpoint_{model_epoch}.pth"
    )

    if cfg.setup.device in ["gpu", "cuda"]:
        checkpoint = torch.load(base_model_dir)
    elif cfg.setup.device == "cpu":
        checkpoint = torch.load(base_model_dir, map_location=torch.device("cpu"))
    else:
        raise ValueError(f"Device {cfg.setup.device} is not admissible")

    model.load_state_dict(checkpoint["model_state_dict"])


def validate_architecture(cfg: DictConfig, logger: logging.Logger) -> bool:
    """
    Validate network architecture and change parameters if needed

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict
    logger: logging.Logger
        module logger
    """
    admissible = True
    dim_img_factor = 2**cfg.network.encoder.len_encoder
    len_filter = 2 * cfg.network.decoder.order_wavelet - 1
    lowest_res_level = (
        cfg.network.decoder.init_res_level - cfg.network.decoder.num_levels_down
    )

    # Compatibility image dimensions and length encoder
    if (
        cfg.dataset.dim_img[0] % dim_img_factor != 0
        or cfg.dataset.dim_img[1] % dim_img_factor != 0
    ):
        logger.warning("2 ** len_encoder is not a factor of image dimension!")
        admissible = False

    # Compatibility lowest resolution level and order wavelet
    if 2**lowest_res_level < len_filter:
        logger.warning("len(filter) > len(signal_lowest_level)!")
        admissible = False

    # Compatibiliy number levels up and initial resolution level
    if cfg.network.decoder.num_levels_up > cfg.network.decoder.num_levels_down:
        logger.warning("num_levels_up is too large!")
        admissible = False

    # Compatibility number levels up and length encoder
    if cfg.network.decoder.num_levels_up > cfg.network.encoder.len_encoder - 1:
        logger.warning("num_levels_up > len_encoder!")
        admissible = False

    # Verify initial reslevel is sufficiently high to cover [-1/2, 1/2]
    if cfg.network.decoder.init_res_level < np.ceil(
        np.log(cfg.network.decoder.order_wavelet) / np.log(2) + 1
    ):
        logger.warning("Initial resolution level is too low: [-1/2,1/2] is not covered")
        admissible = False

    return admissible
