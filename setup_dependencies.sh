#!/bin/bash
# Setup script for improved training dependencies

echo "=== Setting up dependencies for improved training ==="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hiera-seizure

# Install/update required packages
echo "Installing required packages..."

# Core packages
pip install scikit-learn>=1.0.0
pip install scipy>=1.7.0
pip install matplotlib>=3.3.0

# Verify installations
echo "=== Verifying installations ==="
python -c "
import sklearn
import scipy
import matplotlib
from scipy.stats import beta
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print(f'✅ scikit-learn: {sklearn.__version__}')
print(f'✅ scipy: {scipy.__version__}')  
print(f'✅ matplotlib: {matplotlib.__version__}')

# Test beta distribution
test_beta = beta.rvs(4, 4, size=10)
print(f'✅ Beta distribution test: {test_beta[:3]}')

# Test class weight computation
test_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array([0, 0, 1, 1, 1]))
print(f'✅ Class weight test: {test_weights}')

print('All dependencies installed successfully!')
"

echo "=== Creating improved training script ==="
# Create the scripts directory if it doesn't exist
mkdir -p scripts

# The improved training script will be created separately
echo "✅ Dependencies setup complete!"
echo ""
echo "Next steps:"
echo "1. Save the improved training script as 'scripts/improved_training.py'"
echo "2. Save the job script as 'scripts/submit_improved_training.sh'"
echo "3. Run: chmod +x scripts/submit_improved_training.sh"
echo "4. Submit: qsub scripts/submit_improved_training.sh"