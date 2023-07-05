"""Crop set of nrrds using known segmentation"""

import argparse
import SimpleITK as sitk
import numpy as np
import logging
import multiprocessing

from pathlib import Path
from typing import Tuple, Dict

from mra.processing import preprocessing as prep

# Module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def store_array(
    array: np.ndarray, metadata: Dict, export_dir: Path, compression: bool
) -> None:
    """
    Store array as sitk image

    Parameters
    ----------
    array: float-valued np.ndarray of shape [depth heigth width]
        array to be converted
    metadata: Dict
        dictionary containing metadata image
    export_dir: Path
        output dir
    """
    # Construct sitk image
    array_sitk = sitk.GetImageFromArray(array, isVector=False)
    array_sitk.SetSpacing(metadata["spacing"])
    array_sitk.SetDirection(metadata["direction"])
    array_sitk.SetOrigin(metadata["origin"])

    # Export
    sitk.WriteImage(
        array_sitk,
        str(export_dir),
        compression,
    )


def crop_vols(
    img: np.ndarray,
    mask: np.ndarray,
    dim_yx: Tuple[int, int],
    dpix: int,
    resize: prep.Resize,
    min_num_pixels: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image mask pair in depth direction

    Parameters
    ----------
    img: float-valued np.ndarray of shape [depth heigth width]
        image with roi
    mask: binary-valued np.ndarray of shape [depth heigth width]
        binary mask associated to image
    radius: int
        radius to take around center slice
    min_num_pixels: int
        minimal number pixels to classify slice as non-empty

    Returns
    -------
    img_crop: float-valued np.ndarray of shape [depth_crop heigth width]
        image cropped in depth direction
    mask_crop: np.uint8-valued np.ndarray of shape [depth_crop heigth width]
        mask cropped in depth direction
    """
    # Discard empty slices
    non_empty_slices = np.where(np.sum(mask, axis=(1, 2)) >= min_num_pixels)[0]
    num_slices = len(non_empty_slices)
    img_crop_z = img[non_empty_slices]
    mask_crop_z = mask[non_empty_slices]

    # Crop in x,y directions
    img_crop = np.zeros((num_slices, *dim_yx))
    mask_crop = np.zeros((num_slices, *dim_yx), dtype=np.uint8)
    for slice_idx in range(num_slices):
        # Extract measurements mask
        mask_yx_idx = np.where(mask_crop_z[slice_idx] == 1)
        min_y = np.min(mask_yx_idx[0])
        max_y = np.max(mask_yx_idx[0])
        min_x = np.min(mask_yx_idx[1])
        max_x = np.max(mask_yx_idx[1])

        # Extract cropped mask
        mask_slice_bbox = np.uint8(
            mask_crop_z[
                slice_idx, min_y - dpix : max_y + dpix, min_x - dpix : max_x + dpix
            ]
        )
        mask_crop[slice_idx] = resize(np.expand_dims(mask_slice_bbox, 0))[0]

        # Extract cropped image
        img_slice_bbox = img_crop_z[
            slice_idx, min_y - dpix : max_y + dpix, min_x - dpix : max_x + dpix
        ]
        img_crop[slice_idx] = resize(np.expand_dims(img_slice_bbox, 0))[0]

    return img_crop, mask_crop


def crop(args, nrrd_dir: Path, dim_yx: Tuple[int, int], resize: prep.Resize) -> None:
    """
    Crop mask and image in subfolder

    Parameters
    ----------
    nrrd_dir: Path
        folder containing image and mask
    args:
        command line arguments
    """
    logger.info(f"Working on {str(nrrd_dir)}")

    export_dir = args.out_dir / nrrd_dir.relative_to(args.data_dir)
    if not export_dir.exists():
        export_dir.mkdir(parents=True)

    img, metadata = prep.load_img(nrrd_dir / f"{args.img_key}.nrrd", store_meta=True)
    mask = prep.load_img(nrrd_dir / f"{args.mask_key}.nrrd")
    img_crop, mask_crop = crop_vols(img, mask, dim_yx, args.dpix, resize)

    store_array(
        img_crop, metadata, export_dir / f"{args.img_key}.nrrd", args.compression
    )

    store_array(
        mask_crop,
        metadata,
        export_dir / f"{args.mask_key}.nrrd",
        args.compression,
    )


def main():
    """Resize nrrd images in directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", help="path to subfolders containing scans and masks", type=Path
    )
    parser.add_argument("out_dir", help="ouput folder", type=Path)
    parser.add_argument("size", help="desires size", nargs="+", default=[])
    parser.add_argument(
        "--dpix", help="number of pixels around image", type=int, default=15
    )
    parser.add_argument(
        "--num-workers", help="number of processes", type=int, default=4
    )
    parser.add_argument("--img-key", help="image key", type=str, default="scan")
    parser.add_argument("--mask-key", help="mask key", type=str, default="mask")
    parser.add_argument("--compression", help="compress", type=bool, default=True)
    args = parser.parse_args()

    # Initiallization
    out_size = tuple([int(dim_size) for dim_size in args.size])
    nrrd_dirs = list(set([dir.parent for dir in args.data_dir.rglob("*.nrrd")]))
    if not args.out_dir.exists():
        args.out_dir.mkdir()

    # Resize with padding
    resize = prep.Resize(out_size, method="pad")

    # Resize images and export to nrrd
    input = [(args, dir, out_size, resize) for dir in nrrd_dirs]
    with multiprocessing.Pool(args.num_workers) as pool:
        pool.starmap(crop, input)


if __name__ == "__main__":
    main()
