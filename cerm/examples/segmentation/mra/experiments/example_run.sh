#!/bin/bash

conda activate mra

home_dir="/path/to/CERM"
export PYTHONPATH=$PYTHONPATH:${home_dir}/CERM

# Region of interest
roi=prostate

# Set up directories
train_dir="/path/to/train/data"
val_dir="/path/to/val/data"
test_dir="/path/to/test/data"
out_dir="/path/to/out"

python ${home_dir}/cerm/examples/segmentation/main.py --multirun \
	task=$roi \
	setup.train=True \
	setup.num_workers=6 \
	setup.model_dir=${out_dir} \
	dataset.train_dir=${train_dir} \
	dataset.val_dir=${val_dir} \
	dataset.test_dir=${test_dir} \
	hydra.sweep.dir=${out_dir} \
    hydra.sweep.subdir=$roi \
