#!/bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -N test_final_gpu
#$ -o test_final_gpu_output.log

module load cuda/12.1
source /users/akoongbo/miniconda3/etc/profile.d/conda.sh
conda activate hiera-seizure

echo "=== Final GPU Test ==="
nvidia-smi

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.mm(x, x)
    print('✅ GPU computation successful!')
    print('✅ PyTorch CUDA is working perfectly!')
else:
    print('❌ CUDA still not available')
"

cd /users/akoongbo/hiera-seizure-webcam
python test_dataset.py
