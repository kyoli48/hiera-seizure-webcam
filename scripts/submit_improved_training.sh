#!/bin/bash
#$ -M akoongbo@nd.edu
#$ -m abe
#$ -r n
#$ -N improved_seizure_training
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -o improved_training_output.log

echo "=== Improved GESTURE Seizure Detection Training ==="
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
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'NumPy version: {np.__version__}')
print(f'SciPy available: {hasattr(stats, \"beta\")}')
print('✅ All dependencies loaded successfully!')
"

# Change to working directory
cd /users/akoongbo/hiera-seizure-webcam

# Check if data exists
echo "=== Data Check ==="
if [ -d "gestures" ]; then
    echo "✅ Dataset directory found"
    ls -la gestures/
    if [ -f "gestures/seizures.csv" ]; then
        echo "✅ Seizures CSV found"
        head -5 gestures/seizures.csv
    else
        echo "❌ seizures.csv not found"
        exit 1
    fi
else
    echo "❌ Dataset directory not found"
    exit 1
fi

# Run improved training
echo "=== Starting Improved Training ==="
python scripts/improved_training.py

echo "=== Training Complete ==="
echo "Job finished: $(date)"

# Show results
if [ -f "best_model.pth" ]; then
    echo "✅ Best model saved successfully"
    ls -lh best_model.pth
fi

if [ -f "training_curves.png" ]; then
    echo "✅ Training curves saved"
    ls -lh training_curves.png
fi
