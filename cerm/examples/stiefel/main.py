import os
import sys

sys.path.append(os.getcwd()) 

import hydra
import torch
from omegaconf import DictConfig
from typing import Optional
from logging import getLogger
from sklearn.metrics import balanced_accuracy_score

from cerm.examples.stiefel.data.mnist_data import create_mnist_datasets

logger = getLogger(__name__)


@hydra.main(version_base="1.3", config_path="", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets and loaders
    train_dataset, val_dataset, test_dataset = create_mnist_datasets(
        cfg.data.path, cfg.data.random_train_val_split
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )

    # Instantiate model
    logger.info(f"Instantiating network {cfg.network._target_}")
    model = hydra.utils.instantiate(cfg.network).to(device)

    # Instantiate optimizer
    logger.info(f"Instantiating optimizer {cfg.optimizer._target_}")
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    # Instantiate loss
    logger.info(f"Instantiating loss {cfg.loss._target_}")
    loss = hydra.utils.instantiate(cfg.loss)

    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        y_train_true = []
        y_train_pred = []

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_value = loss(output, target)
            loss_value.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()
            y_train_true.extend(target.cpu().numpy())
            y_train_pred.extend(pred)

            total_train_loss += loss_value.item()
            num_train_batches += 1
            
            if batch_idx % cfg.training.log_interval == 0:
                logger.info(f"Epoch: {epoch} | Batch: {batch_idx} | Training loss: {loss_value.item():.4f}")

        avg_train_loss = total_train_loss / num_train_batches
        train_balanced_accuracy = balanced_accuracy_score(y_train_true, y_train_pred)
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        y_val_true = []
        y_val_pred = []

        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss_value = loss(output, target)
                total_val_loss += loss_value.item()
                num_val_batches += 1

                pred = output.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()
                y_val_true.extend(target.cpu().numpy())
                y_val_pred.extend(pred)

        avg_val_loss = total_val_loss / num_val_batches
        val_balanced_accuracy = balanced_accuracy_score(y_val_true, y_val_pred)

        logger.info(
            f"Epoch: {epoch} | Training balanced accuracy: {100. * train_balanced_accuracy:.2f}% | Validation balanced accuracy: {100. * val_balanced_accuracy:.2f}%"
        )
        logger.info(
            f"Epoch: {epoch} | Average Training loss: {avg_train_loss:.4f} | Average Validation loss: {avg_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()