#!/bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -N gesture_training
#$ -o gesture_training_output.log
#$ -M akoongbo@nd.edu
#$ -m abe

echo "=== GESTURE Seizure Detection Training ==="
echo "Job started: $(date)"
echo "Node: $(hostname)"

# Load CUDA module
module load cuda/12.1

# Activate conda environment
source /users/akoongbo/miniconda3/etc/profile.d/conda.sh
conda activate hiera-seizure

# Quick environment check
echo "=== Environment Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Change to working directory
cd /users/akoongbo/hiera-seizure-webcam

# Run training
echo "=== Starting Training ==="
python scripts/train_gesture_hiera.py

echo "=== Training Complete ==="
echo "Job finished: $(date)"
