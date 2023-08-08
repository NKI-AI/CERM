<div align="center">

# CERM: Constrained Empirical Risk Minimization
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

CERM is a deep learning framework for training neural networks with constraints; the implementation details can be found in [REF]. Here we briefly explain how to use the general framework and run the examples accompanied with our NeurIPS 2023 submission "Constrained Empirical Risk Minimization".
 
 </div>

 <br>

## Installation and Usage

Download CERM directly from the repository.
The CERM framework has been tested on PyTorch 2.0.1.

## Using the CERM framework

The CERM framework preserves the usual flow of building models in PyTorch as much as possible. Here we provide a brief example of how one would use our framework to build a model with constraints.

### Using constrained parameters

We provide an abstract class `Constraints` whose implementation should be completed by the user. The user is only required to implement their specific constraint of interest. 
A specific constraint can be applied to different groups of parameters, assuming each group has the same dimensionality. We require the user to provide the following data to the constructor of our `Constraint` class:
>- ***num_params: int***
>    - dimension input of zero map
>- ***num_eqs: int***
>    - number of equations
>- ***num_groups: int***
>    - the same set of equations can be applied to different groups (layers)

Next, we build our model as usual extending the ``torch.nn.Module`` class, 
with the only difference that we use `ConstrainedParameter` instead of `torch.nn.Parameter` in the places where we wish to use constraints.
The constructor of `ConstrainedParameter` requires the following data:

>- `constraint`: an instance of the `Constraint` class.
>- `init_params` (optional): initial guess parameters.

The constructor of the `ConstrainedParameter` will refine the initial guess and constrain it to the constrained manifold. A constrained parameter is 
explicitly constructed using
```python
    from cerm.network.constrained_params import ConstrainedParameter
    constrained_params = ContrainedParameter(constraint=constraint, init_params=params)
```

### Optimizer
Models are trained by using a custom optimizer for the constrained parameters, which 
can be found in `/CERM/cerm/optimizer`, and works like any standard optimizer in PyTorch. 
When writing training routines, one should use the `split_params` function first to split 
the learnable parameteres in the model into constrained and unconstrained ones; see
the example below.
   ```python
    import torch

    from cerm.optimizer.riemannian_sgd import RSGD
    from cerm.network.constrained_params import split_params

    # Split parameters in your model (model is an instance of torch.nn.Module)
    unconstrained_params, constrained_params = split_params(model)

    # Initialize optimizer
    constrained_optimizer = RSGD(unconstrained_params, lr=1e-03)

    # Now use optimizer as usual to update constrained parameters
```

## Examples

### MLP with spherical constraints 
In `/CERM/cerm/examples/sphere` we define a fully connected layer, where the rows of the
weight matrix are constrained to the unit sphere. This example serves only as a minimal toy example
illustrating how to use our framework.
   ```python
    import torch

    from cerm.examples.sphere import spherical_constraints

    # Construct MLP with spherical constraints
    dim_in = 64
    dim_out = 5
    dim_latent = 96
    num_hidden_layers = 4
    mlp = spherical_constraints.MLP(dim_in, dim_latent, dim_out, num_hidden_layers)
    
    # Evaluate 
    bsize = 6
    x = torch.rand(bsize, dim_in)
    y = mlp(x)
```

### MLP constrained to Stiefel manifold
In `/CERM/cerm/examples/stiefel` we define a fully connected layer, where the weight matrix is
constrained to lie on the so-called Stiefel manifold. That is, the columns of the weight matrix form 
an orthonormal set. 
   ```python
    import torch

    from cerm.examples.stiefel import stiefel

    # Initialize Stiefel layer
    dim_in = 10
    dim_out = 15
    stiefel_layer = stiefel.StiefelLayer(dim_in, dim_out, bias=True)
    
    # Apply model
    bsize = 16
    x = torch.rand(bsize, dim_in)
    y = stiefel_layer(x)
```
This setup also contains an example of a domain-specific method to initialize parameters
randomly (points on the Stiefel manifold). 

### Shift equivariant fully connected layer
In `/CERM/cerm/examples/equivariance` we define a fully connected layer which is equivariant to circular shifts.
The purpose of this example is only to illustrate the flexibility of our framework; we know, of course, which
architecture is needed for achieving shift equivariance (CNNs) 
   ```python
    import torch
    import logging

    from torch import Tensor

    from cerm.examples.equivariance.shift import ShiftEquivariantLayer

    # Module logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    def shift(x: Tensor) -> Tensor:
        """Circual shift sequence.
        
        Parameters
        ----------
        x: Tensor
            batch of 1d input (shape [batch_size dim])

        Returns
        -------
        y: Tensor
            input shifted one place to the right
        """
        y = torch.zeros(*x.shape)
        y[:, 0] = x[:, -1]
        y[:, 1:] = x[:, 0:-1]
        return y   
        
    # Initialize shift equivariant layer
    dim = 35
    bsize = 5
    layer = ShiftEquivariantLayer(dim)

    # Test equivariance
    x = 100 * torch.rand(bsize, dim)
    y1 = shift(layer(x))
    y2 = layer(shift(x))

    logger.info(f"Discrepancy equivariance: {torch.max(y1 - y2)}")         
```
### Learnable wavelet layers
A more elaborate example implementing learnable wavelet layers can be found in `/CERM/cerm/examples/wavelets`
In this section we summarize the details of how to use our wavelet-layers using the CERM-framework. The constructor of the 1d wavelet layer requires the following data:

>- `order`: int
>    - order of filter
>- `number_of_decomps`: int
>    - number of levels in wavelet decompositions
>- `num_filters_per_channel`: int (optional)
>    - number of wavelet filters
>- `num_channels`: int (optional)
>    - number of channels in input signal
>- `periodic_signal`: bool (optional)
>    - indicates whether the input signal is periodic

A minimal example using a wavelet layer is given below
```python
    import torch

    from cerm.examples.wavelets.wavelet_layer import WaveletLayer1d

    # Construct learnable wavelet layer
    order = 4 
    num_levels_down = 3
    num_channels = 2
    wavelet_layer = WaveletLayer1d(order, num_levels_down, num_channels=num_channels)

    # Compute decomposition
    bsize = 4
    signal_len = 157
    signal = torch.rand(bsize, num_channels, signal_len)
    approx, detail = wavelet_layer(signal)
```
## Example paper: Autocontouring

The examples from the paper can be run using the supplied `hydra` configs.

### Install dependencies

We provide a conda environment in `/CERM/cerm/examples/mra_segmentation/conda_env.yml` to install the required dependencies:

    conda env create -f /CERM/cerm/examples/mra_segmentation/conda_env.yml

### Data format

For training purposes the scans and associated segmentations need to be stored in `h5` format.

### Preparation of data I: folder structure

A scan and its associated masks need to be stored in one folder. The folder with training data should consist of subfolders containing `nrrd` files. Each subfolder corresponds to a separate scan with associated segmentations. The filenames of a scan and its associated masks need to be identical for each subfolder. For example, the folder setup for a training set may look as follows:
```
    train/
    ├── image_masks_1/
            ├── scan.nrrd
            ├── mask_1.nrrd
            ├── mask_2.nrrd
            ├── mask_3.nrrd
    ├── image_masks_2/
            ├── scan.h5
            ├── mask_1.nrrd
            ├── mask_2.nrrd
            ├── mask_3.nrrd
    ├── ...
    ├── image_masks_n/
            ├── scan.nrrd
            ├── mask_1.nrrd
            ├── mask_2.nrrd
            ├── mask_3.nrrd
```
The folder setup for a validation set should follow the same structure.

### Preparation of data II: convert to h5

After the appropriate folder structures have been set up, the contents of each folder need to be converted to a single `h5` dataset. The names of the scans and masks will be used as keys. After conversion the final folder-structure should be as depicted below:
```
    train/
    ├── image_masks_1/
            ├── scan_with_masks.h5
    ├── image_masks_2/
            ├── scan_with_masks.h5
    ├── ...
    ├── image_masks_n/
            ├── scan_with_masks.h5
```
A script `nrrd_to_h5.py` for performing the conversion to `h5` is available in the tools folder. If the folders with data are structured as prescribed in part I, the following call will set up the required `h5` datasets.
```
    python nrrd_to_h5.py $nrrd_dir $h5_dir
```
### Train models

Models can be trained using the provided configs, e.g., the models for spleen can be trained using
```python
    python main.py --multirun
        task=spleen \
        dataset.train_dir=/path/to/train_dir \
        dataset.val_dir=/path/to/val_dir \
        dataset.test_dir=/path/to/test_dir \
        network.decoder.order_wavelet=3,4,5,6,7,8
```
We refer the reader to `/CERM/cerm/examples/mra_segmentation/mra/configs` for configuration details, and what settings can be overridden, and `/CERM/cerm/examples/mra_segmentation/mra/experiments` for bash scripts containing detailed examples to reproduce the results presented in the paper.