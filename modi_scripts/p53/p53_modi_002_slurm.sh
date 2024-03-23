#!/bin/bash

#SBATCH --job-name P512_002
#SBATCH --nodelist n002
#SBATCH --partition modi_short
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 64
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err


srun apptainer exec ~/modi_images/ucphhpc/hpc-notebook:22.05.11 /home/wnv601_alumni_ku_dk/modi_mount/z_p53_modi_002_mount.sh