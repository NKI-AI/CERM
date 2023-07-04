CERM: constrained empirical risk minimization
=============================================

CERM is a deep learning framework for training neural networks with constraints; 
the implementation details can be found in [REF]_. Here we briefly explain how to 
use the general framework and run the examples accompanied with our NeurIPS 2023 
submission "Constrained Empirical Risk Minimization". 

Installation and Usage
======================

Clone repository
----------------
Download CERM directly from the repository at `anonymous/url` 
or clone from the command-line:

.. code-block:: console

    git clone git@github.com:anonymous/CERM.git

Build conda environment
-----------------------
Build the conda environment using the supplied file in ``/CERM/conda_env.yml``.

.. code-block:: console

    conda env create -f environment.yml

Minimal example - using the CERM framework
==========================================
The CERM framework preserves the usual flow of building models in PyTorch
as much as possible. Here we provide a brief example of how one would use
our framework to build a model with constraints.  

Example: wavelets
==========================================
In this section we summarize the details of how to use our wavelet-layers
using the CERM-framework. 
    
Example paper: Autocontouring
=============================

The examples from the paper can be run using the supplied ``hydra`` configs, e.g., 
the models for the spleen can be trained using 

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

Data format
-----------

For training purposes the scans and associated segmentations need to be
stored in ``h5`` format. If a model is only used to construct (predict) segmentations
the scans may be stored in any format supported by ``sitk``.

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

Bibliography
------------
.. [REF] *REFERENCE / LINK TO PAPER*

