#!/bin/bash

#SBATCH --job-name P512_3
#SBATCH --nodelist n005
#SBATCH --partition modi_short
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 64
#SBATCH -o %x.out
#SBATCH -e %x.err


srun apptainer exec ~/modi_images/ucphhpc/hpc-notebook:22.05.11 /home/wnv601_alumni_ku_dk/modi_mount/C_modi.sh