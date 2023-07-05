"""Module with tools for preprocessing images."""

import numpy as np
import h5py
import torch
import cv2
import SimpleITK as sitk

from torch import Tensor
from pathlib import Path
from skimage import transform
from typing import Tuple


def load_img_h5(file_dir: Path, key_img: str = "img") -> np.ndarray:
    """
    Load image from h5 file.

    Parameters
    ----------
    file_dir: Path
        path to file
    key_img: str
        key to extract image stored in h5 file

    Returns
    -------
    data: hdf5
        image stored in h5 file
    """
    with h5py.File(str(file_dir), "r") as f:
        data = f[key_img][()]
    return data


def load_img(file_dir: Path, store_meta: bool = False) -> np.ndarray:
    """
    Load any sitk supported image into an array.

    Parameters
    ----------
    file_dir: Path
        path to file
    store_meta: bool
        indicates whether to load metadata or not

    Returns
    -------
    float-valued numpy array
        image stored in nrrd file
    """
    img = sitk.ReadImage(f"{str(file_dir)}")
    if store_meta:
        metadata = {
            "filename": str(file_dir),
            "depth": img.GetDepth(),
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
        }
        return sitk.GetArrayFromImage(img), metadata
    else:
        return sitk.GetArrayFromImage(img)


def force_binary(mask: np.ndarray) -> np.ndarray:
    """
    Force mask to be binary. Nonzero values in mask are set to one.

    Parameters
    ----------
    mask: float-valued numpy array
        segmentation with possibly corrupted values

    Returns
    -------
    mask: int-valued array
        segmentation with only ones and zeros (should store as boolean)

    """
    mask.astype(int)
    mask[np.where(mask > 1)] = 1
    return mask


def rescale_intensity(
    target_range: Tuple[float, float],
    array: np.ndarray,
    alpha_min: float = 0.005,
    alpha_max: float = 0.995,
) -> np.ndarray:
    """
    Rescale values array from current array range to prescribed target range.

    Parameters
    ----------
    target_range: 2-tuple of floats
        range to which array-values are to be mapped
    array: float-valued numpy array
        array to be rescaled

    Returns
    -------
    3d float-valued array [num_slices heigth width]
        rescaled array
    """
    x0 = np.quantile(array, alpha_min)
    x1 = np.quantile(array, alpha_max)
    array_clip = np.clip(array, x0, x1)
    if x0 == x1:
        return array_clip
    else:
        return (target_range[1] - target_range[0]) / (x1 - x0) * (
            array_clip - x0
        ) + target_range[0]


class RescaleIntensity(object):
    """Rescale image values from current range to prescribed target range."""

    def __init__(
        self, target_range: Tuple[float, float], scale_per_slice: bool = True
    ) -> None:
        """
        Initialize target range and scaling scope.

        Parameters
        ----------
        target_range: 2-tuple of floats
            range to which array-values are to be mapped
        scale_per_slice: bool
            indicates whether to scale per slice or volume
        """
        self.target_range = target_range
        self.scale_per_slice = scale_per_slice

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Rescale 2d image (min-max scaling)."""
        return rescale_intensity(self.target_range, array)


class ClipIntensity(object):
    """Clip values image."""

    def __init__(self, clip_range: Tuple[float, float]) -> None:
        """
        Initialize clip range.

        Parameters
        ----------
        clip_range: 2-tuple of floats
            interval to which pixel values are clipped
        """
        self.clip_range = clip_range

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Clip image."""
        return np.clip(array, self.clip_range[0], self.clip_range[1])


class Resize(object):
    """Resize image using (linear) interpolation or padding."""

    def __init__(
        self, out_shape: Tuple[int, int], method: str = "pad", interp_order: int = 1
    ) -> None:
        """
        Initialize output shape and order of interpolation.

        Parameters
        ----------
        out_shape: 2-tuple of ints
            desired output shape
        interp_order: int
            interpolation order
        method: str in {pad, interp}
            use either padding or interpolation to resize image
        """
        self.out_shape = out_shape
        self.method = method
        if self.method == "interp":
            self.interp_order = interp_order

    def __call__(self, img):
        """
        Resize image to prescribed output size using polynomial interpolation or padding.

        Parameters
        ----------
        image: float-valued numpy array of size [batch_size heigth width]
            images to be resized

        Returns
        -------
        float-valued numpy array of size [batch_size out_shape[0] out_shape[1]]
            resized images
        """
        if self.method == "interp":
            img = transform.resize(
                np.moveaxis(img, [0, 1, 2], [2, 0, 1]),
                self.out_shape,
                order=self.interp_order,
                preserve_range=True,
            )
            return np.moveaxis(img, [0, 1, 2], [1, 2, 0])

        elif self.method == "pad":
            dim_spat = img.shape[1::]

            for dim_idx in range(len(dim_spat)):
                if self.out_shape[dim_idx] > dim_spat[dim_idx]:
                    num_zeros = self.out_shape[dim_idx] - dim_spat[dim_idx]
                    num_zeros_left = num_zeros // 2
                    pad_width = [(0, 0), (0, 0), (0, 0)]
                    pad_width[dim_idx + 1] = (
                        num_zeros_left,
                        num_zeros - num_zeros_left,
                    )
                    img = np.pad(img, pad_width)
                else:
                    crop_size = dim_spat[dim_idx] - self.out_shape[dim_idx]
                    crop_size_left = crop_size // 2
                    crop_size_right = crop_size_left

                    if crop_size % 2 == 1:
                        crop_size_right = crop_size_right + 1

                    if dim_idx == 0:
                        img = img[
                            :, crop_size_left : dim_spat[dim_idx] - crop_size_right
                        ]
                    else:
                        img = img[
                            ..., crop_size_left : dim_spat[dim_idx] - crop_size_right
                        ]

            return img
