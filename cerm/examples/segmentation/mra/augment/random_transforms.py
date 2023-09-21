"""Module with tools to perform data augmentation."""

import torch
import abc
import random
import torchvision
import torch.nn.functional as nnf

from enum import Enum
from typing import Tuple, List

from mra.augment import transforms, conv
from mra.augment.augment_config_lib import AugmentConfig
from mra.processing import preprocessing


class AugmentModes(str, Enum):
    """Admissible augmentations"""

    shift = "shift"
    rotate = "rotate"
    shear = "shear"
    scale = "scale"
    crop = "crop"
    elastic = "elastic"
    noise = "noise"
    blur = "blur"
    color_jitter = "color_jitter"


class AugmentFactory:
    def __init__(self, cfg: AugmentConfig, dim_img: Tuple[int, int]) -> None:
        """
        Initialize list of transforms which can be applied to (image, mask) pair.

        Parameters
        ----------
        cfg: AugmentConfig
            configuration containing augmentations and parameters
        dim_img: Tuple[int, int]
            dimension image (heigth, width)
        """
        self.augmentations = [Identity()]
        self.num_aug = range(cfg.max_num_aug + 1)
        if cfg.max_num_aug > len(cfg.transforms):
            raise ValueError(
                "Maximal number augmentations should be smaller than number transforms"
            )

        # Construct augmentation list
        dim_img = tuple(dim_img)
        for name in cfg.transforms:
            if name == AugmentModes.shift:
                aug = RandomShift(torch.tensor((cfg.shift.y, cfg.shift.x)), dim_img)
            elif name == AugmentModes.rotate:
                aug = RandomRotate(torch.tensor(cfg.rotate.angle).view(1, 2), dim_img)
            elif name == AugmentModes.shear:
                aug = RandomShear(
                    torch.tensor(cfg.shear.factor).view(1, 2), cfg.shear.axis, dim_img
                )
            elif name == AugmentModes.scale:
                aug = RandomScale(torch.tensor(cfg.scale.factor).view(1, 2), dim_img)
            elif name == AugmentModes.crop:
                aug = RandomCrop(
                    torch.tensor((cfg.crop.midpt_y, cfg.crop.midpt_x)),
                    torch.tensor(cfg.crop.area),
                    dim_img,
                )
            elif name == AugmentModes.elastic:
                aug = RandomDeformation(
                    dim_img,
                    cfg.elastic.grid_field,
                    torch.tensor(cfg.elastic.mean_field),
                    torch.tensor(cfg.elastic.sigma_field),
                    cfg.elastic.sigma_mollifier,
                    cfg.elastic.int_time,
                )
            elif name == AugmentModes.noise:
                aug = RandomNoise(
                    cfg.noise.type, torch.tensor((cfg.noise.loc, cfg.noise.scale))
                )
            elif name == AugmentModes.blur:
                aug = GaussianBlur(torch.tensor((cfg.blur.sigma_y, cfg.blur.sigma_x)))
            elif name == AugmentModes.color_jitter:
                aug = ColorJitter(
                    cfg.color_jitter.brightness,
                    cfg.color_jitter.contrast,
                    cfg.color_jitter.saturation,
                    cfg.color_jitter.hue,
                )
            else:
                raise NotImplementedError(f"{name} has not been implemented")

            self.augmentations.append(aug)

    def __call__(self, sample):
        """
        Apply a random transform to a sample (image, mask) pair.

        Parameters
        ----------
        sample: dictionary containing tensors of size [batch_size num_channels heigth width]
            sample (image, mask) returned by a dataloader of a SingleModal dataset

        Returns
        -------
        sample: dictionary with entries 'image' and 'mask'
            transformed image and mask (in place)
        """
        for _ in range(random.choice(self.num_aug)):
            sample = random.sample(self.augmentations, 1)[0](sample)
        return sample


class RandomParamInit(object):
    """Random parameter initialization."""

    def __init__(self, range_params=(), init_rand_params=None):
        """
        Initialize parameter range and sampling method.

        Parameters
        ----------
        range_params: float-valued Pytorch tensor of size [num_params 2]
            each row corresponds to a prescribed range out which a sample is to be drawn
        init_rand_params: function mapping float-valued Pytorch tensor -> (num_params,) Pytorch
        tensor
            custom defined sampling method (uniform by default)
        """
        self.range_params = range_params
        if not init_rand_params:
            self.init_rand_params = self.init_param_uniform
        else:
            self.init_rand_params = init_rand_params

    @staticmethod
    def init_param_uniform(ran):
        return (ran[:, 1] - ran[:, 0]) * torch.rand(ran.size()[0]) + ran[:, 0]

    def __call__(self):
        """Return random sample from range using prescribed sampling method."""
        if len(self.range_params) > 0:
            return self.init_rand_params(self.range_params)
        else:
            return self.init_rand_params()


class RandomTransform(abc.ABC):
    """2d transformation of (image, mask) pair."""

    def __init__(self, rand_param_init=None, discret_size=None):
        """
        Initialize transforms and grid for randomly transforming images and masks.

        Parameters
        ----------
        rand_param_init: RandomParamInit
            initialization of random parameters (e.g. an angle, direction, etc.)
        discret_size: 2-tuple of ints, optional
            specifies discretization size in each direction
        """
        self.discret_size = discret_size
        if discret_size:
            self.grid = transforms.compute_grid(discret_size)
        else:
            self.grid = None
        self.rand_param_init = rand_param_init

    @abc.abstractmethod
    def transform(self, image, mask, params, grid=None):
        """
        Apply a 2d transform to a collection of (image, mask) pairs.

        Parameters
        ----------
        image: 3d float-valued Pytorch tensor [num_images height width]
            images to be transformed
        mask: 3d float-valued Pytorch tensor [num_classes height width]
            masks to be transformed
        params: float-valued Pytorch tensor of size (num_params, )
            parameters needed to perform transform (e.g. an angle, axis, etc.)
        grid: float-valued Pytorch-tensor of size [n_x n_y 2], optional
            gridpoints {(j,i) : 0 <= j <= n_y - 1, 0 <= x <= n_x - 1}

        Returns
        -------
        2-tuple of 3d float-valued Pytorch tensor [num_images height width]
            transformed (image, mask) pair
        """
        pass

    def __call__(self, sample):
        """
        Apply prescribed transformations to slices of image and channels (classes) of mask.

        Parameters
        ----------
        sample: dictionary with entries 'image' and 'mask' containing tensors [batch_size
        num_channels heigth width]
            sample (image, mask) returned by a dataloader associated to SegData

        Returns
        -------
        sample: dictionary with entries 'image' and 'mask'
            transformed image and mask (in place)
        """
        # Initialize dimensions
        shape_image = sample["image"].size()
        shape_mask = sample["mask"].size()

        # Initialize random parameters
        if self.rand_param_init:
            params = self.rand_param_init()
        else:
            params = None

        # Apply transform
        image, mask = self.transform(
            sample["image"].view(shape_image[0] * shape_image[1], *shape_image[2::]),
            sample["mask"].view(shape_mask[0] * shape_mask[1], *shape_mask[2::]),
            params,
            grid=self.grid,
        )

        # Reshape image and mask to original "batched" shape
        sample["image"] = image.view(*shape_image)
        sample["mask"] = mask.view(*shape_mask)

        return sample


class Identity:
    """Identity transformation."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Apply identity transformation to sample.

        Parameters
        ----------
        sample: dictionary with entries 'image' and 'mask' containing tensors [batch_size
        num_channels heigth width]
            sample (image, mask) returned by a dataloader of a SegData dataset

        Returns
        -------
        sample: dictionary with entries 'image' and 'mask'
            same image and mask pair (in place)
        """
        return sample


class RandomRotate(RandomTransform):
    """Random 2d rotation of (image, mask) pair."""

    def __init__(self, range_angle, discret_size):
        """
        Initialize dimensions and range for rotations.

        Parameters
        ----------
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        range_angle: float-valued Pytorch tensor of size 1 x 2
            interval from which angles are sampled
        """
        super().__init__(
            RandomParamInit(range_params=range_angle), discret_size=discret_size
        )

    @staticmethod
    def inv_rot_mat(angle):
        """
        Construct inverse 2d rotation matrix.

        Parameters
        ----------
        angle: float-valued 1d Pytorch tensor
            angle in radians

        Returns
        -------
        float-valued 2d Pytorch tensor of size 2 x 2
            2d inverse rotation matrix
        """
        return torch.tensor(
            [
                [torch.cos(angle), torch.sin(angle)],
                [-torch.sin(angle), torch.cos(angle)],
            ]
        )

    def transform(self, image, mask, angle, grid):
        """Apply random transformation."""
        return (
            transforms.affine_transformation(
                image, self.inv_rot_mat(angle), mode="bilinear", grid=self.grid
            ),
            transforms.affine_transformation(
                mask, self.inv_rot_mat(angle), mode="nearest", grid=self.grid
            ),
        )


class RandomShift(RandomTransform):
    """Random 2d shift of (image, mask) pair."""

    def __init__(self, range_shift, discret_size):
        """
        Initialize dimensions and range random shift.

        Parameters
        ----------
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        range_shift: float-valued Pytorch tensor of size 2 x 2
            interval from which components of the shift are sampled
        """
        super().__init__(
            RandomParamInit(range_params=range_shift), discret_size=discret_size
        )
        self.identity = torch.tensor([[1, 0], [0, 1]]).float()

    def transform(self, image, mask, direction, grid):
        """Apply random shift."""
        return (
            transforms.affine_transformation(
                image, self.identity, shift=direction, mode="bilinear", grid=self.grid
            ),
            transforms.affine_transformation(
                mask, self.identity, shift=direction, mode="nearest", grid=self.grid
            ),
        )


class RandomShear(RandomTransform):
    """Random 2d shear of (image, mask) pair."""

    def __init__(self, range_shear, axis, discret_size):
        """
        Initialize dimensions and orientation of shear.

        Parameters
        ----------
        range_shear: float-valued Pytorch tensor of size 1 x 2
            interval from which shear factor is sampled
        axis: str in {horizontal, vertical}
            axis along which shear is performed
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        """
        super().__init__(
            RandomParamInit(range_params=range_shear), discret_size=discret_size
        )
        self.axis = axis

    def inv_shear_mat(self, shear_factor):
        """
        Construct inverse shear matrix along prescribed axis.

        Parameters
        ----------
        shear_factor: float-valued 1d Pytorch tensor
            shear factor

        Returns
        -------
        float-valued 2d Pytorch tensor of size 2 x 2
            2d inverse shear matrix
        """
        if self.axis == "horizontal":
            return torch.tensor([[1, -shear_factor], [0, 1]])
        elif self.axis == "vertical":
            return torch.tensor([[1, 0], [-shear_factor, 1]])

    def transform(self, image, mask, shear_factor, grid):
        """Apply random shear."""
        return (
            transforms.affine_transformation(
                image, self.inv_shear_mat(shear_factor), mode="bilinear", grid=self.grid
            ),
            transforms.affine_transformation(
                mask, self.inv_shear_mat(shear_factor), mode="nearest", grid=self.grid
            ),
        )


class RandomScale(RandomTransform):
    """Apply random scale to (image, mask) pair."""

    def __init__(self, range_scale, discret_size):
        """
        Initialize dimensions and scale range.

        Parameters
        ----------
        range_scale: float-valued Pytorch tensor of size 1 x 2
            interval from which scale factor is sampled
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        """
        super().__init__(
            RandomParamInit(range_params=range_scale), discret_size=discret_size
        )
        self.identity = torch.tensor([[1, 0], [0, 1]]).float()

    def transform(self, image, mask, scale, grid):
        """Apply random scale."""
        scale_mat = 1 / scale * self.identity
        return (
            transforms.affine_transformation(
                image, scale_mat, mode="bilinear", grid=self.grid
            ),
            transforms.affine_transformation(
                mask, scale_mat, mode="nearest", grid=self.grid
            ),
        )


class RandomCrop(RandomTransform):
    """Random crop (image, mask) pair."""

    def __init__(self, bbox_pt_range, bbox_prop_area_range, discret_size):
        """
        Initialize dimensions and parameters bounding boxes used for cropping.

        Parameters
        ----------
        bbox_pt_range: 2 x 2 float-valued Pytorch tensor
            range components of point around which image is cropped
        bbox_prop_area_range: 1 x 2 float-valued Pytorch tensor
            range for proportion area that the bounding box should cover
        discret_size: 2-tuple of ints
            dimension size of image [height width]
        """
        super().__init__(
            RandomParamInit(range_params=bbox_pt_range), discret_size=discret_size
        )
        self.bbox_prop_area_range = bbox_prop_area_range
        self.bbox_prop_side_len = torch.sqrt(bbox_prop_area_range)
        self.resize_img = preprocessing.Resize(
            self.discret_size, method="interp", interp_order=1
        )
        self.resize_mask = preprocessing.Resize(
            self.discret_size, method="interp", interp_order=0
        )

    def transform(self, image, mask, bbox_pt, grid):
        """Apply random cropping."""
        bbox = []
        for axis in range(2):
            side_len = self.discret_size[axis] - 1
            pt = random.randint(0, side_len)
            if pt < bbox_pt[axis]:
                right_pt_min = torch.round(pt + self.bbox_prop_side_len[0] * side_len)
                right_pt_max = torch.round(pt + self.bbox_prop_side_len[1] * side_len)
                right_pt = random.randint(right_pt_min, right_pt_max)
                if right_pt > side_len:
                    pt = pt - right_pt + side_len
                    right_pt = side_len
                bbox.append([pt, right_pt])
            else:
                left_pt_min = torch.round(pt - self.bbox_prop_side_len[1] * side_len)
                left_pt_max = torch.round(pt - self.bbox_prop_side_len[0] * side_len)
                left_pt = random.randint(left_pt_min, left_pt_max)
                if left_pt < 0:
                    pt = pt - left_pt
                    left_pt = 0
                bbox.append([left_pt, pt])

        return (
            torch.from_numpy(
                self.resize_img(
                    image[:, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1]].numpy()
                )
            ),
            torch.from_numpy(
                self.resize_mask(
                    mask[:, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1]].numpy()
                )
            ),
        )


class RandomFlip(RandomTransform):
    """Random flip of 2d (image, mask) pair."""

    def __init__(self, range_prob, axis="horizontal", tau=0.5):
        """
        Initialize random flip parameters.

        Parameters
        ----------
        range_prob: float-valued Pytorch tensor of size 1 x 2
            range from which flip probability is sampled
        axis: str in {horizontal, vertical}
            index of axis to flip
        tau: float in (0, 1)
            image is flipped if prob > tau
        """
        super().__init__(RandomParamInit(range_params=range_prob))
        self.axis = axis
        self.tau = tau

    def flip(self, image, prob):
        """
        Flip image along prescribed axis if prob > tau.

        Parameters
        ----------
        image: 3d float-valued Pytorch tensor [num_images height width]
            images to be flipped
        prob: float
            images are flipped if prob > tau

        Returns
        -------
        3d float-valued Pytorch tensor [num_images height width]
            flipped images
        """
        if prob > self.tau:
            return transforms.flip(image, self.axis)
        else:
            return image

    def transform(self, image, mask, prob, grid):
        """Apply random flip."""
        return self.flip(image, prob), self.flip(mask, prob)


class RandomNoise(RandomTransform):
    """Add random noise to 2d (image, mask) pair."""

    def __init__(self, type_noise, dist_params):
        """
        Initialize parameters noise.

        Parameters
        ----------
        type_noise: str in {poisson, gaussian}
            specifies the type of noise
        dist_params: 1d float-valued Pytorch tensor
            the parameters associated to the probability distribution of the noise; [mean var] if
            Gaussian and [lambda] if poisson
        """
        super().__init__()
        self.type_noise = type_noise
        self.dist_params = dist_params
        if type_noise == "poisson":
            self.prob_dist = torch.distributions.poisson.Poisson(self.dist_params[0])
        elif type_noise == "gaussian":
            self.prob_dist = torch.distributions.multivariate_normal.MultivariateNormal(
                self.dist_params[0].view(
                    1,
                ),
                self.dist_params[1].view(1, 1),
            )

    def noise(self, image):
        """
        Add noise to image.

        Parameters
        ----------
        image: 3d float-valued Pytorch tensor [num_images height width]
            images to which noise will be added

        Returns
        -------
        3d float-valued Pytorch tensor [num_images height width]
             images with noise
        """
        image_shape = image.size()
        sample_noise = self.prob_dist.sample(image_shape).view(*image_shape)
        if self.type_noise == "gaussian":
            return image + sample_noise
        elif self.type_noise == "poisson":
            raise NotImplementedError("Poisson noise is not yet implemented")

    def transform(self, image, mask, params, grid):
        """Apply random noise transformation."""
        return self.noise(image), mask


class GaussianBlur(RandomTransform):
    """Blur (image, mask) pair."""

    def __init__(self, range_sigma):
        super().__init__(RandomParamInit(range_params=range_sigma))

    def transform(self, image, mask, sigma, grid):
        """Apply blurring."""
        return transforms.gaussian_blur(image, sigma=sigma), mask


class ColorJitter(RandomTransform):
    """Color jitter on (image, mask) pair."""

    def __init__(self, ran_brightness, ran_contrast, ran_saturation, ran_hue):
        ran_params = torch.tensor(
            (ran_brightness, ran_contrast, ran_saturation, ran_hue)
        )
        super().__init__(RandomParamInit(range_params=ran_params))

    def transform(self, image, mask, jitter_params, grid):
        jitter = torchvision.transforms.ColorJitter(
            brightness=(jitter_params[0], jitter_params[0]),
            contrast=(jitter_params[1], jitter_params[1]),
            saturation=(jitter_params[2], jitter_params[2]),
            hue=(jitter_params[3], jitter_params[3]),
        )
        img_pil = torchvision.transforms.functional.to_pil_image(image)
        return torchvision.transforms.functional.to_tensor(jitter(img_pil)), mask


class RandomGammaCorrection(RandomTransform):
    """Use power law to randomly apply the so-called gamma correction to change illumination."""

    def __init__(self, range_gamma):
        super().__init__(RandomParamInit(range_params=range_gamma))

    def transform(self, image, mask, gamma, grid):
        """Apply random gamma correction."""
        return image**gamma, mask


class RandomGaussianVectorfield(object):
    """Construct random vectorfield by sampling vector components from Gaussian distribution."""

    def __init__(
        self,
        grid_field,
        grid_target,
        mean_field,
        sigma_field,
        kernel=(),
        sigma_mollifier=(),
        interp_mode="bilinear",
        conv_method="toeplitz",
    ):
        """
        Initialize parameters for sampling random vectorfield.

        Parameters
        ----------
        grid_field: 2-tuple of ints
            specifies the grid on which the initial random vectorfield is sampled
        grid_target: 2-tuple of ints
            specifies the grid to which the random vectorfield is to be extended by interpolation
        mean_field: 1d float-valued Pytorch tensor with two components, optional
            mean value Gaussian used for sampling components vectorfield ([mean_field_x
            mean_field_y])
        sigma_field: 1d float-valued Pytorch tensor with two components, optional
            standard deviation Gaussian used for sampling components vectorfield ([sigma_field_x
            sigma_field_y])
        kernel: 2-tuple of 1d Pytorch tensors, optional
            separable kernel
        sigma_mollifier: 1d float-valued Pytorch tensor with two components (needed if no kernel
        is provided)
            level of smoothing (standard deviation Gaussian) in each spatial direction ([sigma_x
            sigma_y])
        interp_mode: str in {bilinear, constant}, optional
            interpolation technique used for upsampling and evaluating vectorfield outside grid
        conv_method: str in {toeplitz, torch, fft}, optional
            convolution method
        """
        self.grid_field = grid_field
        self.grid_target = grid_target
        self.dim_spat = len(grid_field)
        self.mean_field = mean_field.view(self.dim_spat, 1, 1)
        self.sigma_field = sigma_field.view(self.dim_spat, 1, 1)

        if len(kernel) == 0:
            if sigma_mollifier:
                self.kernel = [conv.gaussian_kernel(sigma) for sigma in sigma_mollifier]
                self.sigma_mollifier = sigma_mollifier
            else:
                raise ValueError(
                    "Mollification parameters should be specified if no kernel is \
                                 provided"
                )
        else:
            self.kernel = kernel

        self.interp_mode = interp_mode
        self.conv_method = conv_method

    def __call__(self):
        """
        Construct random vectorfield by sampling components from Gaussian distribution.

        Returns
        -------
        3d float-valued Pytorch tensor of size [heigth width dim_spat]
            randomly sampled (smoothed) vectorfield
        """
        # Construct random vector field and mollify
        # Note: spatial components follow standard geometrical ordering
        vectorfield = self.mean_field + self.sigma_field * torch.randn(
            self.dim_spat, *self.grid_field
        )
        vectorfield.unsqueeze_(0)
        for comp in range(self.dim_spat):
            vectorfield[:, comp] = conv.conv2d_separable(
                vectorfield[:, comp], self.kernel, mode="trunc", method=self.conv_method
            )

        # Interpolate vectorfield on grid on which image is sampled
        vectorfield = nnf.interpolate(
            vectorfield, size=self.grid_target, mode=self.interp_mode
        )
        return vectorfield.view(self.dim_spat, *self.grid_target).permute([1, 2, 0])


class RandomDeformation(RandomTransform):
    """Deform (image, mask) pair using randomly generated (Gaussian) vectorfield."""

    def __init__(
        self,
        discret_size_image,
        grid_field,
        mean_field,
        sigma_field,
        sigma_mollifier,
        int_time,
        interp_mode_field="bilinear",
        interp_mode_image="bilinear",
        conv_method="toeplitz",
        grid=(),
    ):
        """
        Initialize parameters random vectorfield and ODE integrator for elastic deformation.

        Parameters
        ----------
        discret_size_image: 2-tuple of ints
            specifies discretization size of image in each direction
        grid_field: 2-tuple of ints
            specifies the grid on which the initial random vectorfield is sampled
        mean_field: 1d float-valued Pytorch tensor with two components, optional
            mean value Gaussian used for sampling components vectorfield ([mean_field_x
            mean_field_y])
        sigma_field: 1d float-valued Pytorch tensor with two components, optional
            standard deviation Gaussian used for sampling components vectorfield ([sigma_field_x
            sigma_field_y])
        sigma_mollifier: 1d float-valued Pytorch tensor with two components
            level of smoothing (standard deviation Gaussian) in each spatial direction ([sigma_x
            sigma_y])
        int_time: float
            integration time
        interp_mode_field: str in {bilinear, constant}, optional
            interpolation technique used for upsampling and evaluating vectorfield outside grid
        interp_mode_image: str in {bilinear, constant}, optional
            interpolation technique used for evaluating image outside grid
        conv_method: str in {toeplitz, torch, fft}, optional
            convolution method
        """
        random_vectorfield = RandomGaussianVectorfield(
            grid_field,
            discret_size_image,
            mean_field,
            sigma_field,
            sigma_mollifier=sigma_mollifier,
            interp_mode=interp_mode_field,
            conv_method=conv_method,
        )
        super().__init__(
            RandomParamInit(init_rand_params=random_vectorfield),
            discret_size=discret_size_image,
        )
        self.int_time = int_time
        self.interp_mode_image = interp_mode_image

    def transform(self, image, mask, vectorfield, grid):
        """Apply elastic deformation."""
        return (
            transforms.elastic_deformation(
                image,
                vectorfield,
                self.int_time,
                grid=self.grid,
                mode=self.interp_mode_image,
            ),
            transforms.elastic_deformation(
                mask, vectorfield, self.int_time, grid=self.grid, mode="nearest"
            ),
        )
