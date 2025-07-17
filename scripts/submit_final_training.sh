#!/bin/bash
#$ -M akoongbo@nd.edu
#$ -m abe
#$ -r n
#$ -N final_seizure_training
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 2
#$ -cwd
#$ -j y
#$ -o final_training_output.log

echo "=== Final GESTURE Training with Maximum Regularization ==="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $JOB_ID"

# Load required modules
module load python/3.12.11
module load cuda/12.1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hiera-seizure

# Environment check
echo "=== Environment Check ==="
python -c "
import torch
import numpy as np
from scipy import stats
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('✅ Final training environment ready!')
"

# Change to working directory
cd /users/akoongbo/hiera-seizure-webcam

# Run final training
echo "=== Starting Final Training ==="
python scripts/final_training.py

echo "=== Final Training Complete ==="
echo "Job finished: $(date)"

# Show results
if [ -f "best_model_final.pth" ]; then
    echo "✅ Final model saved successfully"
    ls -lh best_model_final.pth
fi

if [ -f "final_training_curves.png" ]; then
    echo "✅ Final training curves saved"
    ls -lh final_training_curves.png
fi

# Compare with previous results
echo "=== Model Comparison ==="
if [ -f "best_model.pth" ]; then
    echo "Previous model size: $(ls -lh best_model.pth | awk '{print $5}')"
fi
if [ -f "best_model_final.pth" ]; then
    echo "Final model size: $(ls -lh best_model_final.pth | awk '{print $5}')"
fi
