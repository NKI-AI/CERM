"""Convert datasets in medicthon dataset to nrrd"""

import argparse
import json
import multiprocessing
import logging
import re
import SimpleITK as sitk
import numpy as np

from pathlib import Path
from typing import List, Dict

# Initialize module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_median_spacing(
    data_dir: Path, metadata: Dict[str, Path], num_dim: int
) -> np.ndarray:
    """Compute median spacing."""
    spacing = []
    for sample in metadata["training"]:
        scan_dir = data_dir / sample["image"]
        spacing_sample = sitk.ReadImage(str(scan_dir)).GetSpacing()
        if num_dim == 4:
            spacing.append(spacing_sample[0:-1])
        else:
            spacing.append(spacing_sample)
    spacing = np.array(spacing)
    return np.mean(spacing, axis=0), np.median(spacing, axis=0)


def resample(
    img_sitk,
    new_spacing: List[int],
    old_spacing: np.ndarray = [],
    fill_value: int = 0,
    is_mask: bool = False,
) -> np.ndarray:
    """Resample images to uniform spacing."""
    dim_spat = np.array(img_sitk.GetSize(), dtype=np.int)
    old_spacing = np.array(img_sitk.GetSpacing())
    new_size = np.round(dim_spat * (old_spacing / new_spacing)).astype(np.int).tolist()

    # Set settings resampler
    resampler = sitk.ResampleImageFilter()

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(img_sitk)


def sample2nrrd(
    sample: str,
    mask_labels: Dict[str, str],
    modality: Dict[str, str],
    num_dim: int,
    data_dir: Path,
    out_dir: Path,
    type: str,
    spacing: np.ndarray,
    compression: bool = True,
) -> None:
    """Convert images in sample to nrrd."""
    # Initialization
    if type == "train":
        scan_dir = data_dir / sample["image"]
        mask_dir = data_dir / sample["label"]
        logger.info(f"Working on {scan_dir} and {mask_dir}")
    elif type == "val":
        scan_dir = data_dir / sample
        logger.info(f"Working on {scan_dir}")

    # Output folder
    nrrd_dir = out_dir / re.sub(".nii", "", scan_dir.stem)
    if not nrrd_dir.exists():
        nrrd_dir.mkdir()

    # Export scans
    if len(spacing) > 0:
        use_resample = True
    else:
        use_resample = False

    scan_sitk = sitk.ReadImage(str(scan_dir))
    if num_dim == 4:
        for key in modality.keys():
            if use_resample:
                scan = resample(scan_sitk[..., int(key)], spacing, is_mask=False)
            else:
                scan = scan_sitk[..., int(key)]
            sitk.WriteImage(scan, str(nrrd_dir / f"{modality[key]}.nrrd"), compression)
    elif num_dim == 3:  # Should correspond to len(modality.keys()) == 1:
        if use_resample:
            scan = resample(scan_sitk, spacing, is_mask=False)
        else:
            scan = scan_sitk
        name = modality["0"]
        sitk.WriteImage(scan, str(nrrd_dir / f"{name}.nrrd"), compression)

    # Export masks
    if type == "train":
        full_mask = sitk.ReadImage(str(mask_dir))
        arr = sitk.GetArrayFromImage(full_mask)

        for key in mask_labels:
            # Extract mask associated to label
            mask = np.zeros(arr.shape, dtype=np.uint8)
            mask[arr == int(key)] = 1

            # Restore spatial properties
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.SetSpacing(full_mask.GetSpacing())
            mask_sitk.SetDirection(full_mask.GetDirection())
            mask_sitk.SetOrigin(full_mask.GetOrigin())

            # Resample and export
            if use_resample:
                mask_sitk = resample(mask_sitk, spacing, is_mask=True)
            sitk.WriteImage(
                mask_sitk,
                str(nrrd_dir / f"{mask_labels[key]}.nrrd"),
                compression,
            )


def clean_label(label):
    """
    Remove unwanted characters from label.

    Parameters
    ----------
    label: str
        name label

    Returns
    -------
        name label without unwanted characters
    """
    label = re.sub(r"\'", "", label)
    label = re.sub(r"\s", "_", label)
    return label


def main():
    """Parse command line input and move (image, mask) pairs to separate folders."""
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", help="path to folder containing (image, mask) pairs", type=Path
    )
    parser.add_argument("out_dir", help="path to output folder", type=Path)
    parser.add_argument("--num-workers", help="number of workers", type=int, default=4)
    parser.add_argument(
        "--spacing",
        help="can be used to resample image to isotropic or median voxel spacing",
        type=str,
        default="original",
    )
    args = parser.parse_args()

    # Construct train and validation folders
    train_dir = args.out_dir / "train"
    val_dir = args.out_dir / "val"
    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)
    if not train_dir.exists():
        train_dir.mkdir()
    if not val_dir.exists():
        val_dir.mkdir()

    # Read metadata
    with open(args.data_dir / "dataset.json") as json_file:
        metadata = json.load(json_file)

    # Mask labels
    mask_labels = metadata["labels"]
    for key in mask_labels:
        mask_labels[key] = clean_label(mask_labels[key])

    # Modality labels
    modality = metadata["modality"]
    for key in modality:
        modality[key] = clean_label(modality[key])

    # Number of dimensions
    if metadata["tensorImageSize"] == "3D":
        num_dim = 3
    elif metadata["tensorImageSize"] == "4D":
        num_dim = 4
    else:
        raise NotImplementedError("Processing only implemented for 3D and 4D data")

    # Compute median spacing over train set
    if args.spacing in ["isotropic", "median"]:
        num_spat_dim = 3
        _, spacing = compute_median_spacing(args.data_dir, metadata, num_dim)
        if args.spacing == "isotropic":  # Dangerous may destroy image!
            spacing = np.min(spacing) * np.ones(num_spat_dim)
    else:
        spacing = []

    # Assume data dir contains two folders, seg and scan, with identical structure
    with multiprocessing.Pool(args.num_workers) as pool:
        input_train = [
            (
                sample,
                mask_labels,
                modality,
                num_dim,
                args.data_dir,
                train_dir,
                "train",
                spacing,
            )
            for sample in metadata["training"]
        ]
        input_val = [
            (
                sample,
                mask_labels,
                modality,
                num_dim,
                args.data_dir,
                val_dir,
                "val",
                spacing,
            )
            for sample in metadata["test"]
        ]
        pool.starmap(sample2nrrd, input_val)
        pool.starmap(sample2nrrd, input_train)


if __name__ == "__main__":
    main()
