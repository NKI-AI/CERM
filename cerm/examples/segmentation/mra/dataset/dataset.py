"""Module for dataset of (image, mask-contour) pairs."""

import torch
import numpy as np
import logging
import h5py

from pathlib import Path
from torch import Tensor
from typing import List, Tuple, Dict, Union
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from mra.processing import preprocessing as prep
from mra.curve.mask_to_curve import contour_from_mask
from mra.utils import bbox


def custom_collate(
    sample: List[Dict[str, Union[Tensor, List[Tensor]]]]
) -> Dict[str, Union[Tensor, List[Tensor]]]:
    """Custome collate: number contour points may differ"""
    contour = [sample[idx].pop("contour") for idx in range(len(sample))]
    sample_collate = default_collate(sample)
    sample_collate["contour"] = contour
    return sample_collate


class ContourData(Dataset):
    """Dataset representing (image, mask-contour) pairs."""

    def __init__(
        self,
        data_dir: Path,
        scan_key: str,
        mask_key: str,
        dim_img: Tuple[int, int],
        init_res_level: int,
        order_fourier: int,
        resize: bool = True,
        min_num_pixels: int = 5,
        bbox_margin: int = 2,
        file_ext: str = "h5",
        preprocess=None,
        transform=None,
    ) -> None:
        """
        Initialize path to data, keywords to find images, masks and contours and processing steps.

        Parameters
        ----------
        data_dir: Path
            path to folder containing subfolders with images and masks
        scan_key: str
            keyword for scan filename
        mask_keys: str
            keywords for mask filename
        dim_img: Tuple[int, int]
            spatial dimensions (height, width) of image slice
        init_res_level: int
            initial resolution from which lower resolution coefficients are constructed
        order_fourier: int
            order of Fourier approximation
        resize: bool
            indicates whether to resize or not
        min_num_pixels: int, optional
            minimal number of pixels used to classify slice as non-empty
        bbox_margin: int
            number of pixels added to height and width of bounding box
        file_ext: str
            file extension of raw data
        preprocess:
            composition of preprocessing mappings applied to image
        transform: DataAug, optional
            list of transforms to randomly choose from (data augmentation)
        """
        super(ContourData, self).__init__()
        self.logger = logging.getLogger(type(self).__name__)

        # Dataset info
        self.data_dir = data_dir
        self.scan_key = scan_key
        self.mask_key = mask_key
        self.file_ext = file_ext
        self.img_paths = list(self.data_dir.rglob(f"*.{self.file_ext}"))

        # Dimensions
        self.dim_img = dim_img
        self.init_res_level = init_res_level
        self.order_fourier = order_fourier

        # Preprocessing
        self.transform = transform
        self.preprocess = preprocess
        self.min_num_pixels = min_num_pixels

        # Bbox scaling factor
        self.bbox_margin = bbox_margin
        self.scale_factor_bbox = 1 / torch.tensor(self.dim_img)

        self.resize = resize
        if resize:
            self.resize_with_pad = prep.Resize(self.dim_img, method="pad")

        # Compute dataset statistics
        self.mean_midpoint = torch.zeros(2, 1)
        self._compute_slice_stats()
        self._log_stats_dataset()

    def _compute_slice_stats(self) -> None:
        """Count slices in volumes and store statistics."""
        self.cum_num_slices = []
        for img_idx, img_path in enumerate(self.img_paths):
            with h5py.File(img_path, "r") as img:
                # Determine number of slices in image
                num_slices = img[self.scan_key].shape[0]
                if img_idx == 0:
                    self.cum_num_slices.append(num_slices)
                else:
                    self.cum_num_slices.append(
                        self.cum_num_slices[img_idx - 1] + num_slices
                    )

                non_empty_slices = np.where(
                    np.sum(img[self.mask_key], axis=(1, 2)) >= self.min_num_pixels
                )[0]

                # Convert slice indices relative to number total number of slices
                if img_idx == 0:
                    self.non_empty_slices = non_empty_slices
                else:
                    self.non_empty_slices = np.concatenate(
                        (
                            self.non_empty_slices,
                            non_empty_slices + self.cum_num_slices[img_idx - 1],
                        )
                    )

        self.cum_num_slices = np.array(self.cum_num_slices)

    def set_mean_midpoint(self, mean_midpoint: Tensor) -> None:
        """Compute mean midpoint contours."""
        self.mean_midpoint = mean_midpoint.view(2, 1)
        self.logger.info(f"Set mean midpoint to {self.mean_midpoint}")

    def _log_stats_dataset(self) -> None:
        """Log gathered statistics dataset."""
        self.logger.info(
            f"Found {len(self.img_paths)} volumes and a total of "
            f"{self.cum_num_slices[-1]} slices \n"
        )

    def set_preprocess(self, preprocess) -> None:
        self.preprocess = preprocess

    def set_data_aug(self, data_aug) -> None:
        self.transform = data_aug

    def img_slice_idx(self, idx: int) -> Tuple[int, int]:
        """
        Determine indices (i, j) associated to j-th slice of the i-th volume.

        Parameters
        ----------
        idx: int
            slice index in stacked ordering

        Returns
        -------
        Tuple[int, int]
            indices (i, j) of i-th image and j-th slice
        """
        if idx < self.cum_num_slices[0]:
            img_idx = 0
            slice_idx = idx
        else:
            img_idx = np.max(np.where(self.cum_num_slices - 1 < idx)[0]) + 1
            slice_idx = idx - self.cum_num_slices[img_idx - 1]

        return img_idx, int(slice_idx)

    def _load_img(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load slice from volume.

        Parameters
        ----------
        idx: int
            slice index in stacked ordering

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            slice associated to (scan, mask) pair
        """
        img_idx, slice_idx = self.img_slice_idx(idx)
        with h5py.File(self.img_paths[img_idx], "r") as image:
            scan = image[self.scan_key][slice_idx]
            mask = prep.force_binary(image[self.mask_key][slice_idx]).astype(float)
        return scan, mask

    def __len__(self) -> int:
        """
        Return number of slices in dataset.

        Returns
        -------
        int
            number of slices in dataset
        """
        return self.cum_num_slices[-1]

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Return (image, mask-coeffs, cont_len) associated to prescribed index idx ~ (i, j).

        Parameters
        ----------
        idx : int
            slice index in stacked ordering

        Returns
        -------
        """
        scan, mask = self._load_img(idx)

        # Preprocess
        if self.preprocess:
            scan = self.preprocess(np.expand_dims(scan, 0))
        if self.resize:
            mask = self.resize_with_pad(np.expand_dims(mask, 0))

        # Construct sample dict to apply random transformation
        sample = {
            "image": torch.from_numpy(scan.astype(np.float32)).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
        }
        if self.transform:
            sample = self.transform(sample)

        # Compute wavelet coefficients contour and bounding box
        (
            is_empty,
            is_corrupt,
            arclen,
            contour,
            contour_dyadic_scaled,
            midpoint_contour,
            midpoint_bbox,
            height_width_bbox,
        ) = contour_from_mask(
            sample["mask"].view(*self.dim_img),
            self.init_res_level,
            self.order_fourier,
            min_num_pix=self.min_num_pixels,
            mean_midpt=self.mean_midpoint,
            bbox_margin=self.bbox_margin,
        )

        # Compute corners points bounding box
        coords_bbox = bbox.midrad_to_coords_bbox(
            midpoint_bbox.unsqueeze(0),
            height_width_bbox.unsqueeze(0),
            torch.tensor([1.0, 1.0]),
        ).squeeze(0)

        sample = {
            "img": sample["image"].view(1, *self.dim_img),
            "mask": sample["mask"].view(1, *self.dim_img),
            "arclen": arclen,
            "contour": contour,
            "contour_dyadic_scaled": contour_dyadic_scaled,
            "midpoint_contour": midpoint_contour,
            "midpoint_bbox": midpoint_bbox,
            "height_width_bbox": height_width_bbox,
            "coords_bbox": coords_bbox,
            "idx": torch.tensor(idx),
            "is_non_empty": torch.tensor(not is_empty).float(),
            "is_corrupt": torch.tensor(is_corrupt),
        }

        return sample
