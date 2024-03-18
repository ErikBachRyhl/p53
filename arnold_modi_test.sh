#!/bin/bash
# Defines where the package should be installed.
# Since the modi_mount directory content is
# available on each node, we define the package(s) to be installed # here so that the node can find it once the job is being executed. export CONDA_PKGS_DIRS=~/modi_mount/conda_dir

# Activate conda in your PATH
# This ensures that we discover every conda environment # before we try to activate it.
source $CONDA_DIR/etc/profile.d/conda.sh

conda activate python3

python3 arnold_parallel.py