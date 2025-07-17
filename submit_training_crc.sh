#!/bin/bash
#$ -M akoongbo@nd.edu
#$ -m abe
#$ -r n
#$ -N gesture_seizure_training
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $JOB_ID"
echo "Working directory: $SGE_O_WORKDIR"

# Change to working directory
cd $SGE_O_WORKDIR

# Load modules
module load python/3.12.11
module load cuda/12.1

# Check environment
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version)"
echo "GPU information:"
nvidia-smi

# Activate conda environment
source ~/.bashrc
conda activate hiera-seizure

# Verify PyTorch can see GPU
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Run the training script
echo "Starting GESTURE seizure detection training..."
python scripts/train_gesture_hiera.py

echo "Job completed at: $(date)"
