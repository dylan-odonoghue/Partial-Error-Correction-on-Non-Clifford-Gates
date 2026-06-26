#!/bin/bash
# setup_env.sh
# Run once on Setonix to create the conda environment.
# Usage: bash setup_env.sh

set -e

export MAMBA_EXE="$MYSCRATCH/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="$MYSCRATCH/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"

# Create environment (Python 3.12, aarch64-compatible)
micromamba create -n qvc python=3.12 -c conda-forge -y
micromamba activate qvc

# Core scientific stack
pip install numpy scipy

# PennyLane + GPU backend (aarch64 wheels now available on PyPI)
pip install pennylane
pip install pennylane-lightning-gpu

# NVIDIA cuQuantum (required by lightning.gpu)
# Use cu12 for CUDA 12.x (Setonix GH200 uses CUDA 12)
pip install cuquantum-python-cu12

# Data loading
pip install scikit-learn Pillow

echo "Environment setup complete."
echo "Test with: python -c \"import pennylane as qml; print(qml.about())\""
