"""Create h5 dataset."""

import argparse
import h5py
import json
import multiprocessing
import time
import logging
import numpy as np

from pathlib import Path

from mra.processing import preprocessing

# Set up module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def img_to_h5(data_dir: Path, out_dir: Path) -> None:
    """
    Store images, masks and in a subfolder in h5 dataset.

    Parameters
    ----------
    data_dir: Path
        path to subfolder containing one scan, associated masks and contour representations
    out_dir: Path
        path to output folder
    """
    start_time = time.perf_counter()
    logger.info(f"Working on images and masks in {data_dir}.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir, "w") as h5_file:
        # Append scan and masks stored in nrrds
        for dir in data_dir.glob("*.nrrd"):
            image, metadata = preprocessing.load_img(dir, store_meta=True)
            if image.dtype == np.uint8:
                compression = True
            else:
                compression = False
            h5_file.create_dataset(
                dir.stem, data=image, dtype=image.dtype, compression=compression
            )
            h5_file.create_dataset(f"metadata_{dir.stem}", data=json.dumps(metadata))

        # Append arrays stored in numpy files
        for dir in data_dir.glob("*.npy"):
            with open(dir, "rb") as numpy_file:
                array = np.load(numpy_file, allow_pickle=True)
                h5_file.create_dataset(
                    dir.stem, data=array, dtype=array.dtype, compression=False
                )

    time_taken = time.perf_counter() - start_time
    logger.info(f"Time taken for {data_dir}: {time_taken:.3f} seconds.")


def main():
    """Store scan, masks and contour representations in h5 format."""
    # Set up parser
    parser = argparse.ArgumentParser(
        description="Convert directory to H5 file format for H5 file reader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Directory to dataset.", type=Path)
    parser.add_argument("output", help="Directory to write HDF5 to.", type=Path)
    parser.add_argument("--num-workers", default=8, help="Number of workers", type=int)
    parser.add_argument(
        "--dataset-description",
        default=None,
        type=Path,
        help="JSON file describing data structure",
    )
    args = parser.parse_args()

    # Locate scans and masks
    if args.dataset_description:
        raise NotImplementedError(f"Not yet implemented.")
    else:
        subdirs = list(set([dir.parent for dir in args.input.rglob("*.nrrd")]))
        input = [
            (dir, args.output / dir.relative_to(args.input) / "scan_with_masks.h5")
            for dir in subdirs
        ]

    # Convert to h5
    with multiprocessing.Pool(args.num_workers) as pool:
        pool.starmap(img_to_h5, input)


if __name__ == "__main__":
    main()
