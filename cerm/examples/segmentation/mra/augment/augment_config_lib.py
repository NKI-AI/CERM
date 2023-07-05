"""Data augmentation structured config"""

import numpy as np

from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, List


@dataclass
class Shift:
    x: Tuple[float, float] = (-48.0, 48.0)
    y: Tuple[float, float] = (-48.0, 48.0)


@dataclass
class Rotate:
    angle: Tuple[float, float] = (-0.1, 0.1)


@dataclass
class Shear:
    axis: str = "horizontal"
    factor: Tuple[float, float] = (-0.05, 0.05)


@dataclass
class Scale:
    factor: Tuple[float, float] = (0.75, 1.3)


@dataclass
class Crop:
    midpt_x: Tuple[float, float] = (128, 128)
    midpt_y: Tuple[float, float] = (128, 128)
    area: Tuple[float, float] = (0.85, 0.95)


@dataclass
class Elastic:
    grid_field: Tuple[int, int] = (128, 128)
    mean_field: Tuple[float, float] = (0.0, 0.0)
    sigma_field: Tuple[float, float] = (5.0, 5.0)
    sigma_mollifier: Tuple[float, float] = (1.65, 1.65)
    int_time: float = 1.0


@dataclass
class Noise:
    type: str = "gaussian"
    loc: float = 0.0
    scale: float = 5e-05


@dataclass
class Blur:
    sigma_x: Tuple[float, float] = (0.55, 0.95)
    sigma_y: Tuple[float, float] = (0.55, 0.95)


@dataclass
class ColorJitter:
    brightness: Tuple[float, float] = (0.0, 1.0)
    contrast: Tuple[float, float] = (0.0, 1.0)
    saturation: Tuple[float, float] = (0.0, 1.0)
    hue: Tuple[float, float] = (-0.5, 0.5)


@dataclass
class AugmentConfig:
    shift: Shift = Shift()
    rotate: Rotate = Rotate()
    scale: Scale = Scale()
    shear: Shear = Shear()
    crop: Crop = Crop()
    noise: Noise = Noise()
    blur: Blur = Blur()
    elastic: Elastic = Elastic()
    color_jitter: ColorJitter = ColorJitter()
    max_num_aug: int = 4
    transforms: Tuple[str] = ()
