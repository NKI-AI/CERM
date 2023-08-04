"""Main script for training models."""

import logging
import torch
import random
import numpy as np
import hydra
import os

from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy
from torch import Tensor
from tensorflow.python.summary.summary_iterator import summary_iterator
from omegaconf import DictConfig

from mra.dataset import data_utils
from mra.processing.prep_utils import init_preprocessing
from mra.network import model_utils
from mra.augment.random_transforms import AugmentFactory
from mra.optimizer.train_utils import init_training
from mra.utils import inference, stats, bbox, timing
from mra.pnp_hyperparam import process_score


def init_logger(out_dir: Path) -> logging.Logger:
    """
    Set up logger

    Parameters
    ----------
    out_dir: Path
        directory in which output logger is stored

    Returns
    -------
    logger: logging.Logger
        module logger
    """
    logger = logging.getLogger(__name__)
    return logger


@hydra.main(config_path="mra/configs", config_name="train.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Train model or use one to predict."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # If model output folder not specified use current directory
    if "model_dir" not in cfg.setup:
        cfg.setup.model_dir = os.getcwd()

    # Set up device (no multipu for now)
    if cfg.setup.device == "gpu":
        cfg.setup.device = str(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # Set up main directories and initialize logger
    out_dir = Path(cfg.setup.model_dir)
    out_stats = out_dir / "stats"
    if not out_stats.exists():
        out_stats.mkdir()
    logger = init_logger(out_dir)
    # data_utils.slurm_copy_data(cfg, logger)

    # Check if architecture is valid
    if not model_utils.validate_architecture(cfg, logger):
        raise ValueError("Invalid architecture!")

    # Initialize preprocessing steps and augmentations
    preprocess = init_preprocessing(cfg)
    if len(cfg.augment.transforms) > 0:
        data_aug = AugmentFactory(cfg.augment, cfg.dataset.dim_img)
    else:
        data_aug = []

    # Datasets
    train_data = data_utils.init_dataset(
        cfg, Path(cfg.dataset.train_dir), preprocess, data_aug=data_aug
    )
    val_data = data_utils.init_dataset(
        cfg, Path(cfg.dataset.val_dir), deepcopy(preprocess)
    )
    test_data = data_utils.init_dataset(
        cfg, Path(cfg.dataset.test_dir), deepcopy(preprocess)
    )

    # logger.info("Compute mean midpoint contours for centering")
    # stats.init_stats_dataset(cfg, out_stats, train_data, val_data, test_data)

    # Model
    model = model_utils.init_model(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params} ({num_params / 10.0 ** 3:.2f}k).")

    if cfg.setup.train:
        mean_val_loss = init_training(cfg, out_dir, model, train_data, val_data)
        dice_score = float(mean_val_loss["dice"])
        process_score(cfg, dice_score, hash(cfg))
        return dice_score

    if cfg.setup.export_model_performance or cfg.setup.predict:
        model_utils.load_checkpoint(cfg, model)

    if cfg.setup.export_model_performance:
        for mode, dataset in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            logger.info(f"Export model performance on {mode} dataset")
            stats.model_performance(
                model,
                dataset,
                cfg.trainer.dataloader.batch_size,
                out_stats / mode,
                cfg.setup.device,
                prefetch_factor=cfg.trainer.dataloader.prefetch_factor,
                num_workers=cfg.setup.num_workers,
            )

    if cfg.setup.predict:
        for mode, dataset in [("val", val_data), ("test", test_data)]:
            logger.info(f"Export contours {mode} dataset")
            inference.export_predictions(
                model,
                dataset,
                out_dir / "predictions" / mode,
                num_workers=cfg.setup.num_workers,
                device=cfg.setup.device,
                batch_size=cfg.trainer.dataloader.batch_size,
            )

    if cfg.setup.export_wavelets:
        logger.info("Export initial and final learned wavelets")
        init_epoch = 0
        final_epoch = cfg.setup.epoch
        inference.export_wavelets(
            cfg, model, init_epoch, final_epoch, out_dir / "wavelets"
        )

    if cfg.setup.measure_inference_speed:
        logger.info("Measure inference speed")
        timing.measure_inference_speed(
            model,
            out_dir / "inference_speed",
            batch_size=cfg.trainer.dataloader.batch_size,
            device=cfg.setup.device,
        )


if __name__ == "__main__":
    main()
