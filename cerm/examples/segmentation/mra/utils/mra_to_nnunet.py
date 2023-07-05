"""Convert mra folder structure to nn-unet folder structure"""

import SimpleITK as sitk
import argparse
import yaml
import json
import numpy as np
import multiprocessing
import logging

from pathlib import Path
from typing import List, Dict

# Initialize module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def export_metadata(
    task_dir: Path,
    task_name: str,
    modality: str,
    mask_keys: List[str],
    nrrds_train: List[Path],
    nrrds_val: List[Path],
    train_data: List[Dict],
    val_data: List[Dict],
) -> None:
    """
    Export metadata to json as prescribed by NNUnet.

    Parameters
    ----------
    task_dir: Path
        dir to output
    task_name: str
        name of project / task
    modality: str
        modality of scan
    mask_keys: List[str]
        mask keys
    nrrds_train: List[Path]
        directories containing masks and scans for training using mra folder convention
    nrrds_val: List[Path]
        directories containing masks and scans for validation using mra folder convention
    train_data: List[Dict[Path]]
        list of dictionaries containing paths of (scan, mask) pairs for training
    val_data: List[Dict[Path]]
        list of dictionaries containing paths of (scan, mask) pairs for validation
    """
    metadata = {}
    metadata["name"] = task_name
    metadata["description"] = "mra dataset converted to NNUnet format"
    metadata["reference"] = "Empty"
    metadata["license"] = "Empty"
    metadata["release"] = "Empty"
    metadata["tensorImageSize"] = "3D"
    metadata["modality"] = {"0": f"{modality}"}
    metadata["labels"] = {}
    metadata["numTraining"] = len(nrrds_train)
    metadata["numTest"] = len(nrrds_val)
    metadata["training"] = train_data
    metadata["test"] = val_data
    for class_idx, mask_key in enumerate(mask_keys):
        metadata["labels"][str(class_idx)] = mask_key

    with open(task_dir / "dataset.json", "w") as jsonfile:
        json.dump(metadata, jsonfile)


def convert_nrrd(
    idx: int,
    dir: Path,
    task_name: str,
    scan_dir: Path,
    label_dir: Path,
    img_key: str,
    mask_keys: List[str],
    compress: bool = True,
    zero_fill_id: int = 3,
) -> None:
    """
    Move and convert nrrd at prescribed dir to NNUnet folder.

    Parameters
    ----------
    idx: int
        index associated to nrrd
    dir: Path
        path to nrrd file
    task_name: str
        name of project
    scan_dir: Path
        NNUnet directory scans
    label_dir: Path
        NNUnet directory for masks
    img_key: str
        image key
    mask_keys: List[str]
        mask keys
    """
    # TODO: retrieve and store pixel-spacing
    logging.info(f"Working on {dir}")
    scan = sitk.ReadImage(str(dir / f"{img_key}.nrrd"))
    base_name = f"{task_name}_{str(idx).zfill(zero_fill_id)}"
    scan_name = f"{base_name}_0000.nii.gz"
    sitk.WriteImage(scan, str(scan_dir / scan_name), compress)

    if label_dir:
        mask_name = f"{base_name}.nii.gz"

        # Construct mask
        mask_list = [
            sitk.GetArrayFromImage(sitk.ReadImage(str(dir / f"{key}.nrrd")))
            for key in mask_keys[1:]  # Assume first key is background
        ]
        mask_arr = np.zeros(mask_list[0].shape)

        for idx, mask in enumerate(mask_list):
            mask_arr[np.where(mask == 1)] = idx + 1

        # Write mask to disk with metadata
        mask_sitk = sitk.GetImageFromArray(mask_arr)
        mask_sitk.CopyInformation(scan)
        sitk.WriteImage(mask_sitk, str(label_dir / mask_name), compress)


def parse_dir(
    nrrd_dirs: List[Path],
    task_dir: Path,
    mode: str,
    img_key: str,
    mask_keys: List[str],
    task_name: str,
    num_workers: int,
    zero_fill_id: int = 3,
) -> List[Dict]:
    """
    Convert nrrds to NNUnet format.

    Parameters
    ----------
    nrrd_dir: List[Path]
        directories containing masks and scans using mra folder convention
    task_dir: Path
        directory to which nrrds are converted using NNUnet folder convention
    mode: str in {train, val}
        specifies whether the nrrds are used for training or validation,
    img_key: str
        image key
    mask_keys: List[str]
        list of mask keys
    task_name: str
        name of project
    zero_fill_id: int
        number of zeros to fill ids with
    num_workers: int
        number of processes

    Returns
    -------
    img_mask_dirs: List
        List containing relative paths of images and associated masks
    """
    img_mask_dirs = []

    if mode == "train":
        scan_dir = task_dir / "imagesTr"
        label_dir = task_dir / "labelsTr"
    elif mode == "val":
        scan_dir = task_dir / "imagesTs"
        label_dir = []

    # Move and convert images
    input = [
        (idx, dir, task_name, scan_dir, label_dir, img_key, mask_keys)
        for idx, dir in enumerate(nrrd_dirs)
    ]
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(convert_nrrd, input)

    # Store relative paths of images and associated masks
    for idx, dir in enumerate(nrrd_dirs):
        base_name = f"{task_name}_{str(idx).zfill(zero_fill_id)}"
        scan_name = f"{base_name}.nii.gz"

        if mode == "train":
            mask_name = f"{base_name}.nii.gz"
            img_mask_dirs.append(
                {"image": f"./imagesTr/{scan_name}", "label": f"./labelsTr/{mask_name}"}
            )

        elif mode == "val":
            img_mask_dirs.append(f"./imagesTs/{scan_name}")

    return img_mask_dirs


def main():
    """Convert mra folder structure to (MSD) folder structure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_dir", help="directory containing train nrrd files", type=Path
    )
    parser.add_argument(
        "val_dir", help="directory containing test nrrd files", type=Path
    )
    parser.add_argument("out_dir", help="output where results are stored", type=Path)
    parser.add_argument(
        "config_file", help="path to json file containing scan and mask keys", type=str
    )
    parser.add_argument("modality", help="Modality", type=str)
    parser.add_argument("--task-name", help="name of task", default="Task", type=str)
    parser.add_argument("--task-id", help="Task id", default="500", type=str)
    parser.add_argument("--num-workers", help="Number of workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    img_key = config["keys"]["image_key"]
    mask_keys = ["background"] + config["keys"]["mask_keys"]

    # Set up global folder structure
    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)
    task_dir = args.out_dir / f"Task{args.task_id}_{args.task_name}"
    train_dir = task_dir / "imagesTr"
    test_dir = task_dir / "imagesTs"
    label_dir = task_dir / "labelsTr"
    for dir in [args.out_dir, task_dir, train_dir, test_dir, label_dir]:
        if not dir.exists():
            dir.mkdir()

    # Train data
    nrrds_train = list(set([dir.parent for dir in args.train_dir.rglob("*.nrrd")]))
    train_data = parse_dir(
        nrrds_train,
        task_dir,
        "train",
        img_key,
        mask_keys,
        args.task_name,
        args.num_workers,
    )

    # Validation data
    nrrds_val = list(set([dir.parent for dir in args.val_dir.rglob("*.nrrd")]))
    val_data = parse_dir(
        nrrds_val, task_dir, "val", img_key, mask_keys, args.task_name, args.num_workers
    )

    export_metadata(
        task_dir,
        args.task_name,
        args.modality,
        mask_keys,
        nrrds_train,
        nrrds_val,
        train_data,
        val_data,
    )


if __name__ == "__main__":
    main()
