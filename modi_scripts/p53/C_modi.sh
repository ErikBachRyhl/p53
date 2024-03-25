#!/bin/bash

source $CONDA_DIR/etc/profile.d/conda.sh
modi-load-environments
conda activate test

python3 C_python.py