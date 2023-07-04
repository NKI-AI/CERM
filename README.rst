.. default-role:: math

CERM: constrained empirical risk minimization
=============================================

CERM is a deep learning framework for training neural networks with constraints; 
the implementation details can be found in [REF]_. Here we briefly explain how to 
use the general framework and run the examples accompanied with our NeurIPS 2023 
submission "Constrained Empirical Risk Minimization". 

Installation and Usage
======================

The CERM framework has been tested on PyTorch 2.0.1. 

Download CERM directly from the repository at `anonymous/url` 
or clone from the command-line:

.. code-block:: console

    git clone git@github.com:anonymous/CERM.git

Minimal example - using the CERM framework
==========================================
The CERM framework preserves the usual flow of building models in PyTorch
as much as possible. Here we provide a brief example of how one would use
our framework to build a model with constraints.  

Using constrained parameters
----------------------------
We provide an abstract class ``Constraints`` whose implementation should
be completed by the user. The user is only required to implement their
specific constraint of interest. To efficiently deal with such cases, i.e., 
apply one constraint to different groups of parameters, 
where each group has the same dimensionality, we require the user to provide 
the following data to the constructor of our ``Constraint`` class:
    num_params: int
        dimension input of zero map
    num_eqs: int
        number of equations

Next, we build our model as usual, with the only difference that we use
``ConstrainedParameter`` instead of ``torch.nn.Parameter`` where our 
parameters are constrained. The constructor of ``ConstrainedParameter``
requires the following data:
    - ``constraint``: an instance of the ``Constraint`` class. 
    - ``init_params`` (optional): initial guess parameters. 
The constructor of the ``ConstrainedParameter`` will refine the initial
guess and constrain it to the constrained manifold. A constrained parameter
is explicitly constructed using 

.. code-block:: python

    constrained_params = ContrainedParameter(constraint=constraint, init_params=params)


A simple toy example can be found in ``/CERM/cerm/examples/spherical_constraint.py``
A more elaborate examples implemented learnable wavelet layers can be found in 
``/CERM/cerm/examples/wavelets.py``

Example: wavelets
==========================================
In this section we summarize the details of how to use our wavelet-layers
using the CERM-framework. The constructor of the 1d wavelet layer requires
the following data:
    - order: int
        order of filter
    - number_of_decomps: int
        number of levels in wavelet decompositions
    - num_filters_per_channel: int (optional)
        number of wavelet filters
    - num_channels: int (optional)
        number of channels in input signal
    - periodic_signal: bool (optional)
        indicates whether the input signal is periodic

A minimal example using a wavelet layer is given below

.. code-block:: python

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


    
Example paper: Autocontouring
=============================
The examples from the paper can be run using the supplied ``hydra`` configs. 

Install dependencies
--------------------
We provide a conda environment in ``/CERM/cerm/examples/mra_segmentation/conda_env.yml``
to install the required dependencies:

.. code-block:: console

    conda env create -f /CERM/cerm/examples/mra_segmentation/conda_env.yml

Data format
-----------

For training purposes the scans and associated segmentations need to be
stored in ``h5`` format. 

Preparation of data I: folder structure
---------------------------------------
A scan and its associated masks need to be stored in one folder. The folder
with training data should consist of subfolders containing ``nrrd`` files. Each subfolder
corresponds to a separate scan with associated segmentations. The filenames
of a scan and its associated masks need to be identical for each subfolder.
For example, the folder setup for a training set may look as follows:

.. code-block:: none

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

The folder setup for a validation set should follow the same structure.

Preparation of data II: convert to h5
-------------------------------------
After the appropropriate folder structures have been set up, the contents of each
folder need to be converted to a single ``h5`` dataset. The names of the scans
and masks will be used as keys. After conversion the final folder-struture should
be as depicted below:

.. code-block:: none

  train/
    ├── image_masks_1/
            ├── scan_with_masks.h5
    ├── image_masks_2/
            ├── scan_with_masks.h5
    ├── ...
    ├── image_masks_n/
            ├── scan_with_masks.h5

A script ``nrrd_to_h5.py`` for performing the conversion to ``h5`` is available in the tools folder.
If the folders with data are structured as prescribed in part I, the following call will 
set up the required ``h5`` datasets.

.. code-block:: console
  
  python nrrd_to_h5.py $nrrd_dir $h5_dir

Train models
-------------
Models can be trained using the provided configs, e.g., the models for spleen can be
trained using

.. code-block:: console

    python main.py --multirun 
        task=spleen \
        dataset.train_dir=/path/to/train_dir \
        dataset.val_dir=/path/to/val_dir \
        dataset.test_dir=/path/to/test_dir \
        network.decoder.order_wavelet=3,4,5,6,7,8          

We refer the reader to ``/CERM/cerm/examples/mra_segmentation/mra/configs``
for configuration details, and what settings can be overridden, and 
``/CERM/cerm/examples/mra_segmentation/mra/experiments`` for bash scripts
containing detailed examples to reproduce the results presented in the paper. 

Bibliography
------------
.. [REF] *REFERENCE / LINK TO PAPER*

