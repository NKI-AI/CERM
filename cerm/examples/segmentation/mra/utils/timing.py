"""Tools for visualizing predictions."""

import torch
import numpy as np
import logging

from pathlib import Path

# Set up module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def measure_inference_speed(
    model: torch.nn.Module,
    out_dir: Path,
    batch_size: int = 1,
    device: str = "cpu",
    num_warmup: int = 64,
    num_time: int = 512,
) -> None:
    """
    Estimate inference time.

    Parameters
    ----------
    model: ConoutrModel (subclass of nn.Module)
        neural network modelling approximation and detail coefficients of contour
    logger: Logger
        logger to which progress is written
    out_dir: Path
        output path to which results are written
    batch_size: int, optional
        number of samples used in each iteration
    device: str
        device to be used (cpu or gpu)
    num_workers: int
        number of processes used in multiprocessing step for exporting images
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Initialization
    model.eval()

    # Dimensions
    num_channels = 1
    img_size = model.dim_in
    input = torch.rand(batch_size, num_channels, *img_size).to(device)

    # Warmup
    logger.info(f"Warming up GPU ...")
    for _ in range(num_warmup):
        model(input)

    # Meausure inference time
    times = np.zeros((num_time))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    logger.info(f"Meausure inference time ...")
    with torch.no_grad():
        for time_idx in range(num_time):
            # Inference
            start.record()
            y = model(input)
            end.record()
            # Wait for synchronization
            torch.cuda.synchronize()
            times[time_idx] = start.elapsed_time(end)

    mean_time = np.mean(times)
    std_time = np.std(times)

    # Export
    with open(out_dir / "inference_time.txt", "w") as out:
        out.write(f"Mean inference time: {mean_time} \n")
        out.write(f"Std inference time: {std_time}")
