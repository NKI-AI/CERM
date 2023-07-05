"""Convert ACDC dataset to nrrd."""

import argparse
import re
import logging
import numpy as np
import h5py
import multiprocessing
import SimpleITK as sitk
import os
import nibabel as nib

from pathlib import Path
from typing import List, Tuple


# Set up module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dimension and mask labels
permute_dim = {"in": [0, 1, 2], "out": [1, 2, 0]}
labels = {
    0: "background",
    1: "ventricular_cavity_r",
    2: "myocardium",
    3: "ventricular_cavity_l",
}


def filenames(folder: Path) -> List[str]:
    """Extract names of frames and patient"""
    patient = folder.stem
    frame_list = []
    for gz_file in list(folder.rglob("*.gz")):
        frame = re.findall("frame\d+", gz_file.stem)
        if frame:
            frame_list.append(frame[0])
    return patient, list(set(frame_list))


def nii_to_numpy(
    patient: str, frame_list: List[str], folder: Path
) -> Tuple[np.ndarray]:
    """Load images and masks from nii and store in numpy array."""
    img_list = []
    mask_list = []
    for frame in frame_list:
        img = nib.load(str(folder / f"{patient}_{frame}.nii.gz")).get_fdata()
        mask = nib.load(str(folder / f"{patient}_{frame}_gt.nii.gz")).get_fdata()
        img_list.append(np.moveaxis(img, permute_dim["in"], permute_dim["out"]))
        mask_list.append(np.moveaxis(mask, permute_dim["in"], permute_dim["out"]))
    return np.concatenate(img_list, axis=0), np.concatenate(mask_list, axis=0)


def numpy_to_nrrd(
    img: np.ndarray,
    mask: np.ndarray,
    scan_key: str,
    out_dir: Path,
    compression: bool = True,
) -> None:
    """Export numpy arrays to h5 datasets."""
    # Export scan
    sitk.WriteImage(
        sitk.GetImageFromArray(img),
        str(out_dir / f"{scan_key}.nrrd"),
        compression,
    )

    for label_idx, label in labels.items():
        mask_class = np.zeros(mask.shape, dtype=np.uint8)
        mask_class[mask == label_idx] = 1
        sitk.WriteImage(
            sitk.GetImageFromArray(mask_class),
            str(out_dir / f"{label}.nrrd"),
            compression,
        )


def init_argparse() -> None:
    """Initialize command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert ACDC dataset to nrrd",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_dir", type=Path, help="path to images and masks")
    parser.add_argument("out_dir", type=Path, help="output dir")
    parser.add_argument("--scan-key", type=str, help="scan key", default="scan")
    return parser.parse_args()


def main():
    args = init_argparse()

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    subfolders = [
        Path(folder.path) for folder in os.scandir(args.data_dir) if folder.is_dir()
    ]

    for folder in subfolders:
        logging.info(f"Working on {folder}")

        out_dir_h5 = args.out_dir / folder.stem
        if not out_dir_h5.exists():
            out_dir_h5.mkdir(parents=True)

        patient, frame_list = filenames(folder)
        img, mask = nii_to_numpy(patient, frame_list, folder)
        numpy_to_nrrd(img, mask, args.scan_key, out_dir_h5)


if __name__ == "__main__":
    main()
