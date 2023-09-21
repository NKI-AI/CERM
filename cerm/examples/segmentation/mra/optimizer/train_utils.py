"""Train utilities"""

import torch

from torch import Tensor
from omegaconf import DictConfig
from pathlib import Path
from typing import Dict

from mra.network.model import ContourModel
from mra.dataset.dataset import ContourData
from mra.metrics.loss import ContourLoss
from mra.optimizer import trainer


def init_loss(
    cfg: DictConfig, mean_midpoint: Tensor, pretrain: bool = False
) -> ContourLoss:
    """
    Initialize contour loss function

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict
    mean_midpont: float-valued PyTorch Tensor of shape [2]
        average midpoint contours
    pretrain: bool
        incorporate active contour loss if false
    """
    if pretrain:
        weights = {
            "approx_high_l2": cfg.loss.approx_high_l2,
            "active_contour": 0.0,
        }
    else:
        weights = {
            "approx_high_l2": cfg.loss.approx_high_l2,
            "active_contour": cfg.loss.active_contour,
        }

    return ContourLoss(
        weights,
        mean_midpoint,
        cfg.network.decoder.init_res_level,
    )


def init_training(
    cfg: DictConfig,
    out_dir: Path,
    model: ContourModel,
    train_data: ContourData,
    val_data: ContourData,
) -> Dict[str, float]:
    """
    Initialize training model; to be cleaned up

    Parameters
    ----------
    cfg: DictConfig
        hydra configuration dict
    model: ContourModel (inherited class of nn.Module)
        neural network modeling wavelet decomposition of contour
    train_data: ContourData
        training data
    val_data: ContourData
        validation data
    loss: ContourLoss
        loss function

    Returns
    -------
    mean_val_loss: Dict[str, float]
        mean performance on validation set
    """
    loss_pretrain = init_loss(cfg, train_data.mean_midpoint, pretrain=False)

    post_warmup = trainer.Trainer(
        model, loss_pretrain, train_data, val_data, cfg.setup.device, out_dir
    )

    if cfg.setup.resume:
        base_model_dir = Path(cfg.trainer.base_model_dir)
    else:
        if cfg.trainer.warmup_epochs > 0:
            init_epochs = cfg.trainer.warmup_epochs

            # Quick and dirty for now (to be cleaned up)
            warmup_scheduler_encoder = (
                lambda epoch: (cfg.trainer.encoder.lr - cfg.trainer.encoder.init_lr)
                / init_epochs
                * epoch
                / cfg.trainer.encoder.init_lr
                + 1
            )
            warmup_scheduler_decoder = (
                lambda epoch: (cfg.trainer.decoder.lr - cfg.trainer.decoder.init_lr)
                / init_epochs
                * epoch
                / cfg.trainer.decoder.init_lr
                + 1
            )

            warmup = trainer.Trainer(
                model, loss_pretrain, train_data, val_data, cfg.setup.device, out_dir
            )

            warmup.train(
                cfg.trainer.encoder,
                cfg.trainer.decoder,
                cfg.trainer.dataloader.batch_size,
                cfg.trainer.warmup_epochs,
                num_workers=cfg.setup.num_workers,
                slices_with_mask=cfg.trainer.dataloader.slices_with_mask,
                prefetch_factor=cfg.trainer.dataloader.prefetch_factor,
                save_epoch=True,
                save_freq=init_epochs - 1,
                monitor_freq=1,
                warmup=True,
                schedule_method={
                    "encoder": warmup_scheduler_encoder,
                    "decoder": warmup_scheduler_decoder,
                },
            )

            base_model_dir = (
                out_dir / "checkpoints" / f"model_checkpoint_{init_epochs - 1}.pth"
            )
        else:
            base_model_dir = []

        if cfg.trainer.pretrain_epochs > 0:
            epochs = cfg.trainer.warmup_epochs + cfg.trainer.pretrain_epochs

            post_warmup.train(
                cfg.trainer.encoder,
                cfg.trainer.decoder,
                cfg.trainer.dataloader.batch_size,
                epochs,
                slices_with_mask=cfg.trainer.dataloader.slices_with_mask,
                num_workers=cfg.setup.num_workers,
                prefetch_factor=cfg.trainer.dataloader.prefetch_factor,
                save_epoch=True,
                save_freq=epochs - 1,
                monitor_freq=cfg.trainer.monitor_freq,
                base_model_dir=base_model_dir,
                warmup=False,
            )

            base_model_dir = (
                out_dir / "checkpoints" / f"model_checkpoint_{epochs - 1}.pth"
            )

    # Train with full loss
    post_warmup.loss = init_loss(cfg, train_data.mean_midpoint, pretrain=False)
    mean_val_loss = post_warmup.train(
        cfg.trainer.encoder,
        cfg.trainer.decoder,
        cfg.trainer.dataloader.batch_size,
        cfg.trainer.epochs,
        slices_with_mask=cfg.trainer.dataloader.slices_with_mask,
        num_workers=cfg.setup.num_workers,
        prefetch_factor=cfg.trainer.dataloader.prefetch_factor,
        save_epoch=True,
        save_freq=cfg.trainer.save_freq,
        monitor_freq=cfg.trainer.monitor_freq,
        base_model_dir=base_model_dir,
        warmup=False,
    )

    return mean_val_loss
