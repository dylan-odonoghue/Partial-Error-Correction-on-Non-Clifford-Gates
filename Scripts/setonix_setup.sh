#!/bin/bash
# setonix_setup.sh
# Run once on Setonix to create the virtual environment on top of the PyTorch container.
# Usage: bash setonix_setup.sh

set -e

# Load the PyTorch module
module load pytorch/2.7.1-rocm6.3.3

# Get the container path from the module
echo "Using container: $SINGULARITY_CONTAINER"

# Create the virtual environment inside the container
# --system-site-packages inherits PyTorch and other container packages
singularity exec $SINGULARITY_CONTAINER \
    python3 -m venv --system-site-packages \
    /scratch/pawsey1116/dodonoghue/Partial-Error-Correction-on-Non-Clifford-Gates/.partial_qec_venv

# Install additional packages into the venv inside the container 
# Using pennylane==0.38.0 and autoray==0.6.11 for compatibility with the PyTorch container's version of Numpy
singularity exec $SINGULARITY_CONTAINER \
    bash -c "source /scratch/pawsey1116/dodonoghue/Partial-Error-Correction-on-Non-Clifford-Gates/.partial_qec_venv/bin/activate \
    && pip install autoray==0.6.11 pennylane==0.38.0"

echo "Setup complete. Verify with:"
echo "module load pytorch/2.7.1-rocm6.3.3"
echo "singularity exec \$SINGULARITY_CONTAINER bash -c 'source /scratch/pawsey1116/dodonoghue/Partial-Error-Correction-on-Non-Clifford-Gates/.partial_qec_venv/bin/activate && python3 -c \"import pennylane; print(pennylane.__version__)\"'"