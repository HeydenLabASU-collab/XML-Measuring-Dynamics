#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH -t 0-00:60:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load Mamba
module load mamba/latest

# Activate environment
source activate MLPipeline

python3 run.py --system VGSDWRFLRGYHQYQ --replica 1 --label 0
python3 run.py --system VGSDWRFLRGYHQYA --replica 1 --label 1
