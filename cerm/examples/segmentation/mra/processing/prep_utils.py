"""Preprocessing utilities"""

import torchvision

from omegaconf import DictConfig

from mra.processing import preprocessing as prep


def init_preprocessing(cfg: DictConfig) -> torchvision.transforms.Compose:
    """
    Initialize preprocessing steps.

    Parameters
    ----------
    cfg: dict
        hydra configuration dict

    Returns
    -------
    preprocess: Torchvision compose object
        composition of preprocessing functions
    """
    preprocess = []

    if cfg.prep.clip_range:
        preprocess.append(prep.ClipIntensity(cfg.prep.clip_range))

    if cfg.prep.intensity_range:
        preprocess.append(prep.RescaleIntensity(cfg.prep.intensity_range))

    if cfg.prep.resize:
        preprocess = [prep.Resize(cfg.dataset.dim_img, method="pad")] + preprocess

    return torchvision.transforms.Compose(preprocess)
