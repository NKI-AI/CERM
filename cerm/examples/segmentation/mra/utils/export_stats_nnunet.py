"""Quick and dirty script to export nnunet results"""

import argparse
import logging
import json
import numpy as np
import torch
import re
import SimpleITK as sitk

from typing import List, Tuple
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff

# Initialize module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_dice(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    eps: float = 1e-06,
    spatial_dim: Tuple[int, int] = (1, 2),
) -> List[float]:
    """
    Compute 2d dice score

    Parameters
    ----------
    mask_gt: float-valued numpy array of shape [num_slices heigth width]
        groundtruth mask
    mask_pred: float-valued numpy array of shape [num_slices heigth width]
        predicted mask

    Returns
    -------
    dice: List[float]
        dice scores per slice
    """
    card_intersect = np.sum(mask_pred * mask_gt, axis=spatial_dim)
    card_pred_obs = np.sum(mask_pred, axis=spatial_dim) + np.sum(
        mask_gt, axis=spatial_dim
    )
    dice_scores = (2 * card_intersect + eps) / (card_pred_obs + eps)
    return dice_scores.tolist()


def compute_hausdorff(mask_gt: np.ndarray, mask_pred: np.ndarray) -> List[float]:
    """
    Compute 2d hausdorff distance between masks

    Parameters
    ----------
    mask_gt: float-valued numpy array of shape [num_slices heigth width]
        groundtruth mask
    mask_pred: float-valued numpy array of shape [num_slices heigth width]
        predicted mask

    Returns
    -------
    hausdorff: List[float]
        hausdorff scores per slice
    """
    hausdorff = []
    for slice_pred, slice_gt in zip(mask_pred, mask_gt):
        coords_pred = np.stack(np.where(slice_pred == 1), axis=1)
        coords_gt = np.stack(np.where(slice_gt == 1), axis=1)
        dist_pred_gt = directed_hausdorff(coords_pred, coords_gt)[0]
        dist_gt_pred = directed_hausdorff(coords_gt, coords_pred)[0]
        hausdorff.append(np.max((dist_pred_gt, dist_gt_pred)))
    hausdorff = [h for h in hausdorff if h < np.inf]
    return hausdorff


def main():
    """Export results nnUNet"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", help="directory containing test or validation set", type=Path
    )
    parser.add_argument("out_dir", help="output where results are stored", type=Path)
    parser.add_argument("--task-name", help="name of task", default="Task", type=str)
    parser.add_argument("--num-workers", help="Number of workers", type=int, default=4)
    parser.add_argument(
        "--min-num-pixels", help="minimal number positive pixels", type=int, default=5
    )
    args = parser.parse_args()

    # Assume only one modality
    modality_uid = "0000"
    file_ext = "nii.gz"

    with open(args.data_dir / "dataset.json", "r") as json_file:
        json_dict = json.load(json_file)

    num_non_empty_slices = 0
    dice = []
    hausdorff = []
    for sample in json_dict["training"]:
        # Filename without modality uid (for some reason nnUNet does not export those)
        filename = re.findall(f"{args.task_name}_\d+", sample["label"])[0]

        # Load masks
        gt_dir = args.data_dir / "labelsTr" / f"{filename}.{file_ext}"
        pred_dir = args.data_dir / "pred" / f"{filename}.{file_ext}"
        mask_gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_dir)))
        mask_pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_dir)))

        # Determine slices with non-empty mask
        non_empty_slices = np.where(
            np.sum(mask_gt, axis=(1, 2)) >= args.min_num_pixels
        )[0]
        mask_gt = mask_gt[non_empty_slices]
        mask_pred = mask_pred[non_empty_slices]

        # Scores
        dice += compute_dice(mask_gt, mask_pred)
        hausdorff += compute_hausdorff(mask_gt, mask_pred)

    # Export scores
    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)
        torch.save(torch.tensor(dice), args.out_dir / f"dice.pth")
        torch.save(torch.tensor(hausdorff), args.out_dir / f"hausdorff.pth")
    else:
        raise FileExistsError("Directory already exists!")


if __name__ == "__main__":
    main()
