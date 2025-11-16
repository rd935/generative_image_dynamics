#!/bin/bash
#SBATCH --job-name=frame_synth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/frame_synth-%j.out
#SBATCH --error=logs/frame_synth-%j.err
#SBATCH --mail-user=rd935@rutgers.edu
#SBATCH --mail-type=END,FAIL

# Go to project directory
cd /scratch/rd935/generative_image_dynamics

# Make sure logs directory exists
mkdir -p logs

# Load conda and activate env
conda activate /scratch/$USER/conda/envs/gid

echo "Running on: $(hostname)"
nvidia-smi

# Launch training
python train_frame_synthesis.py

