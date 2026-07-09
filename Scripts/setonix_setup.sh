#!/bin/bash
# setonix_setup.sh
# Run once on Setonix to create the python virtual environment in your working directory.
# Usage: bash setonix_setup.sh

set -e 

# Ensure correct python module is loaded
module load python/3.11.6

# Create a virtual environment in the current directory
python -m venv .partial_qec_venv

# Load the virtual environment
source .partial_qec_venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy==2.4.6 scipy==1.17.1 mpi4py==1.4.2 pennylane==0.45.0 pennylane-lightning==0.45.0 torch torchvision mpi4py