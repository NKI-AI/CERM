"""Module with tools for training a model."""

import os
import torch
import logging
import time
import random
import numpy as np

from pathlib import Path
from typing import List, Dict, Tuple, Union, Callable
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from mra.dataset.dataset import ContourData
from mra.dataset.data_utils import construct_data_loader
from mra.metrics.loss import l2_norm_grad_loss, mean_loss, ContourLoss
from mra.network.model import ContourModel
from mra.utils.inference import move_sample_to_device, remove_corrupt_samples
from mra.utils.plot_utils import plot_contours_tensorboard, plot_wavelets_tensorboard

from cerm.optimizer.riemannian_sgd import RSGD
from cerm.network.constrained_params import split_params


class Trainer:
    """Class for training segmentation model."""

    def __init__(
        self,
        model: ContourModel,
        loss: ContourLoss,
        train_data: ContourData,
        val_data: ContourData,
        device: str,
        out_dir: Path,
        metrics: List = None,
    ) -> None:
        """
        Initialize training specifications for model and set up folder structure.

        Parameters
        ----------
        model: ContourModel (subclass of nn.Module)
            neural network modeling Fourier coefficients of contour
        loss: ContourLoss
            loss function
        train_data: ContourData
            training data
        val_data: ContourData
            validation data
        optim_method: str in {RAdam, Adam, SGD}
            optimization method
        device: str
            device to be used (cpu or gpu)
        out_dir: Path
            path to folder where results will be stored
        schedule_method: Python function, optional
            scheduler to update learning rate as function of epoch
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.loss = loss
        self.train_data = train_data
        self.val_data = val_data
        self.metrics = metrics
        self.out_dir = out_dir
        self.device = device

        self.checkpoint_dir = out_dir / "checkpoints"
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        self.prediction_dir = out_dir / "predictions"
        if not self.prediction_dir.exists():
            self.prediction_dir.mkdir()

        self.tensorboard_dir = out_dir / "tensorboard"
        if not self.tensorboard_dir.exists():
            self.tensorboard_dir.mkdir()

    def save_checkpoint(
        self,
        model: ContourModel,
        optimizer_encoder: torch.optim,
        optimizer_decoder: torch.optim,
        epoch: int,
        filename: str = "model_checkpoint",
        ext: str = "pth",
        del_prev_epoch: bool = False,
    ) -> None:
        """
        Save current checkpoint and optionally delete previous one.

        Parameters
        ----------
        model: ContourModel (subclass of nn.Module)
            neural network modeling Fourier coefficients of contour
        optimizer: optim
            optimizer used for training
        epoch: int
            most recent (current) epoch to save
        filename: str
            filename of checkpoint
        ext: str
            extension of checkpoint file
        del_prev_epoch: bool
            indicates whether previous checkpoint should be deleted or not
        """
        self.logger.info(f"Epoch {epoch} - Store model")
        path_out = self.checkpoint_dir / f"{filename}_{epoch}.{ext}"

        if path_out.exists():
            self.logger.warning(
                f"{filename}_{epoch}.{ext} already exists and will be overwritten!"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_encoder_state_dict": optimizer_encoder.state_dict(),
                "optimizer_decoder_state_dict": optimizer_decoder.state_dict(),
            },
            path_out,
        )

        if epoch >= 1:
            if del_prev_epoch:
                path_out_prev = self.checkpoint_dir / f"{filename}_{epoch - 1}.{ext}"
                if path_out_prev.exists():
                    self.logging.info(f"\t Remove model epoch {epoch - 1}")
                    os.remove(self.checkpoint_dir / f"{filename}_{epoch - 1}.{ext}")

    def init_optim(
        self,
        model_params: List[torch.nn.Parameter],
        optim_config: Dict[str, Union[float, str]],
        custom_schedule_method: Callable,
        warmup: bool,
        constrained: bool = False,
    ) -> Tuple[torch.optim.Optimizer, StepLR]:
        """
        Initialize model parameters (if prescribed), metrics and optimizer.

        Parameters
        ----------
        optim_config: Dict[str, Union[float, str]]
            optimizer and scheduler configuration encoder
        model_params: List[torch.nn.Parameter]
            model parameters associated to optimizer
        warmup: bool
            indicates whether optimizer is used for warmup or not
        constrained: bool
            indicates whether constrained optimizer is used or not

        Returns
        -------
        optimizer: torch.optim.Optimizer
            pytorch wrapper around optimizer
        scheduler: torch.optim.lr_scheduler.StepLR
            learning rate scheduler
        """
        if warmup:
            lr = optim_config.init_lr
            if constrained:
                method = "RSGD"
            else:
                method = "SGD"
            momentum = 0
        else:
            lr = optim_config.lr
            method = optim_config.method
            momentum = optim_config.momentum

        # Initialize optimizer
        if method == "Adam":
            optimizer = torch.optim.Adam(
                model_params,
                lr=lr,
                weight_decay=optim_config.weight_decay,
            )
        elif method == "Radam":
            optimizer = torch.optim.RAdam(
                model_params,
                lr=lr,
                weight_decay=optim_config.weight_decay,
            )
        elif method == "SGD":
            optimizer = torch.optim.SGD(
                model_params,
                lr=lr,
                momentum=momentum,
                weight_decay=optim_config.weight_decay,
            )
        elif method == "RSGD":
            optimizer = RSGD(model_params, lr=lr)
        else:
            raise NotImplementedError(f"{method} is not implemented")

        # if base_model_dir:
        #    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Set up scheduler (todo: should be specified by user not hardcoded)
        if custom_schedule_method:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, custom_schedule_method
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=optim_config["scheduler"]["decay_factor"],
                threshold=optim_config["scheduler"]["threshold"],
                patience=optim_config["scheduler"]["patience"],
                verbose=True,
            )

        return optimizer, scheduler

    def update_lr(self, scheduler: StepLR, mean_val_loss: float, warmup: bool) -> None:
        """Update learning rate using custom schedule method or reduce on Plateau."""
        if warmup:
            scheduler.step()
        else:
            scheduler.step(mean_val_loss)

    def init_train_val_loaders(
        self,
        batch_size: int,
        slices_with_mask: bool,
        num_workers: int,
        prefetch_factor: int,
    ) -> Tuple[int, DataLoader, int, DataLoader]:
        """
        Initialize dataloaders

        Parameters
        ----------
        batch_size: int
            batch size
        slices_with_mask: bool
            indicates whether to consider only samples with masks during training
        num_workers: int
            number of workers
        prefetch_factor: int
            number of samples preloaded by each worker

        Returns
        ------
        num_train_samples: int
            number of train samples in train dataset
        num_val_samples: int
            number of validation samples in validation dataset
        train_loader: DataLoader
            dataloader training set
        val_loader: DataLoader
            dataloader validation set
        """
        if slices_with_mask:
            slice_idx_train = self.train_data.non_empty_slices
            slice_idx_val = self.val_data.non_empty_slices
        else:
            slice_idx_train = None
            slice_idx_val = None

        # Data loaders: validation and train
        num_train_samples, train_loader = construct_data_loader(
            self.train_data,
            batch_size,
            num_workers,
            subsample_idx=slice_idx_train,
            prefetch_factor=prefetch_factor,
        )
        num_val_samples, val_loader = construct_data_loader(
            self.val_data,
            batch_size,
            num_workers,
            subsample_idx=slice_idx_val,
            shuffle=False,
            prefetch_factor=prefetch_factor,
        )

        return num_train_samples, num_val_samples, train_loader, val_loader

    def init_monitor_loaders(
        self, num_samples_monitor: int, batch_size: int, num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        """
        Initialize dataloaders for monitoring samples (visualization)

        Parameters
        ----------
        num_samples_monitor: int
            number of samples to monitor during training
        batch_size: int
            batch size
        num_workers: int
            number of workers

        Returns
        ------
        monitor_loader: Dict[str, DataLoader]
            dictionary of dataloaders for train and validation set
        """
        monitor_loader = {}

        train_monitor_slices = np.random.choice(
            self.train_data.non_empty_slices,
            size=num_samples_monitor,
            replace=False,
        )
        val_monitor_slices = np.random.choice(
            self.val_data.non_empty_slices, size=num_samples_monitor, replace=False
        )

        for key, subsample_idx, dataset in zip(
            ["train", "val"],
            [train_monitor_slices, val_monitor_slices],
            [self.train_data, self.val_data],
        ):
            _, monitor_loader[key] = construct_data_loader(
                dataset,
                batch_size,
                num_workers,
                subsample_idx=subsample_idx,
                shuffle=False,
                persistent_workers=True,
            )

        return monitor_loader

    def write_mean_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_train_samples: int,
        num_val_samples: int,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate mean metrics and export to tensorboard.

        Parameters
        ----------
        writer: SummaryWriter,
            writer to tensorboard
        epoch: int
            current epoch
        train_loader: DataLoader
            dataloader for training set
        val_loader: DataLoader
            dataloader vor validation set
        num_train_samples: int
            number of samples in training set
        num_val_samples: int
            number of samples in validation set

        Returns
        -------
        mean_train_loss: Dict[str, float]
            mean performance over train set
        mean_val_loss: Dict[str, float]:
            mean performance over validation set
        """
        # Training loss
        logging.info(f"Epoch {epoch} - Compute train loss")
        start_time = time.time()
        mean_train_loss = mean_loss(self.model, self.loss, train_loader, self.device)
        complete_train_loss = mean_train_loss["complete_loss"]
        logging.info(f"Epoch {epoch} - Mean train loss = {complete_train_loss}")
        logging.info(f"\t \t time_elapsed = {time.time() - start_time}")

        # Validation loss
        logging.info(f"Epoch {epoch} - Compute validation loss")
        start_time = time.time()
        mean_val_loss = mean_loss(self.model, self.loss, val_loader, self.device)
        complete_val_loss = mean_val_loss["complete_loss"]
        logging.info(f"Epoch {epoch} - Mean validation loss = {complete_val_loss}")
        logging.info(f"\t \t time_elapsed = {time.time() - start_time}")

        # Write to tensorboard
        for loss_key in mean_train_loss:
            writer.add_scalars(
                f"{loss_key}/val",
                {"val": mean_val_loss[loss_key], "train": mean_train_loss[loss_key]},
                epoch,
            )

        return mean_train_loss, mean_val_loss

    def train(
        self,
        encoder_optim_config: Dict[str, Union[float, str]],
        decoder_optim_config: Dict[str, Union[float, str]],
        batch_size: int,
        epochs: int,
        base_model_dir: Path = None,
        num_workers: int = 4,
        save_epoch: bool = True,
        save_freq: int = 4,
        monitor_freq: int = 1,
        slices_with_mask: bool = True,
        num_samples_monitor: int = 8,
        prefetch_factor: int = 3,
        warmup: bool = False,
        schedule_method=None,
    ) -> Dict[str, float]:
        """
        Train a neural network.

        Parameters
        ----------
        encoder_optim_config: Dict[str, Union[float, str]]
            optimizer and scheduler configuration encoder
        decoder_optim_config: Dict[str, Union[float, str]]
            optmizer and scheduler configuration decoder
        batch_size: int
            number of samples used in each iteration
        epochs: int
            number of epochs
        base_model_dir: Path, optional
            path of previous checkpoint to be resumed
        num_workers: int
            number of processes used for multiprocessing
        save_epoch: bool
            store all epochs if true
        slices_with_mask: bool
            indicates whether to consider only samples with masks during training

        Returns
        -------
        mean_val_loss: Dict[str, float]
            mean performance on validation set
        """
        writer = SummaryWriter(self.tensorboard_dir)
        if not schedule_method:
            schedule_method = {"encoder": None, "decoder": None}

        # Initialize dataloaders for training and validation
        (
            num_train_samples,
            num_val_samples,
            train_loader,
            val_loader,
        ) = self.init_train_val_loaders(
            batch_size, slices_with_mask, num_workers, prefetch_factor
        )

        # Initialize dataloaders to monitor samples (for visualization)
        if num_samples_monitor > 0:
            monitor_loader = self.init_monitor_loaders(
                num_samples_monitor, batch_size, num_workers
            )

        # Resume training if base model is specified
        if base_model_dir:
            checkpoint = torch.load(base_model_dir)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            init_epoch = checkpoint["epoch"] + 1
        else:
            init_epoch = 0

        # (Re)load optimizer
        encoder_params, decoder_params = split_params(self.model)
        optimizer_encoder, scheduler_encoder = self.init_optim(
            encoder_params,
            encoder_optim_config,
            schedule_method["encoder"],
            warmup,
            constrained=False,
        )
        optimizer_decoder, scheduler_decoder = self.init_optim(
            decoder_params,
            decoder_optim_config,
            schedule_method["decoder"],
            warmup,
            constrained=True,
        )

        for epoch in range(init_epoch, epochs):
            self.model.train()
            iteration = 0
            mean_grad_loss = 0
            mean_train_loss = 0

            for sample in train_loader:
                start_time_iter = time.time()
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()

                # Train on accurate groundtruth only
                remove_corrupt_samples(sample)
                if len(sample["img"]) == 0:
                    continue

                # Predict and evaluate loss
                move_sample_to_device(sample, self.device)
                pred = self.model(sample["img"])
                loss_eval = self.loss(sample, pred)["complete_loss"]
                loss_eval.backward()

                with torch.no_grad():
                    optimizer_encoder.step()
                    optimizer_decoder.step()

                # Keep track of gradient size and estimate average
                with torch.no_grad():
                    curr_batch_size = sample["img"].shape[0]
                    norm_grad_loss = l2_norm_grad_loss(self.model)
                    mean_grad_loss += (
                        curr_batch_size * norm_grad_loss / num_train_samples
                    )

                # Log metrics; todo: move to proper function
                logging.info(
                    f"Epoch {epoch} - iter {iteration} - batch_loss = {loss_eval}"
                )
                logging.info(f"\t \t \t ||grad_w_L|| = {norm_grad_loss}")
                logging.info(f"\t \t \t time_elapsed = {time.time() - start_time_iter}")

                iteration += 1

            # Approximate average loss (not the average since weights are changing)
            logging.info(
                f"Epoch {epoch} - Mean ||grad_w_L|| (train) = {mean_grad_loss}"
            )

            # A posteriori analysis: mean validation loss
            self.model.eval()
            writer.add_scalar("loss/train/mean_grad", mean_grad_loss, epoch)
            mean_train_loss, mean_val_loss = self.write_mean_metrics(
                writer,
                epoch,
                train_loader,
                val_loader,
                num_train_samples,
                num_val_samples,
            )
            self.update_lr(scheduler_encoder, mean_val_loss["complete_loss"], warmup)
            self.update_lr(scheduler_decoder, mean_val_loss["complete_loss"], warmup)

            # Store current model
            if save_epoch and epoch % save_freq == 0:
                self.save_checkpoint(
                    self.model, optimizer_encoder, optimizer_decoder, epoch
                )

            if num_samples_monitor > 0 and epoch % monitor_freq == 0:
                # Plot learned wavelets and refinement masks
                plot_wavelets_tensorboard(
                    writer,
                    self.model,
                    epoch,
                )

                # Plot predicted wavelet decompositions
                for data_type in monitor_loader:
                    plot_contours_tensorboard(
                        writer,
                        self.model,
                        monitor_loader[data_type],
                        self.train_data.mean_midpoint,
                        data_type,
                        epoch,
                        self.device,
                    )

        writer.close()
        return mean_loss(self.model, self.loss, val_loader, self.device)
