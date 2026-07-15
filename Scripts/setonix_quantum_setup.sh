#!/bin/bash

set -e

module purge
module load pawsey pawseytools pawseyenv PrgEnv-nvidia craype-arm-grace
module load gcc-native-mixed/12.3
module load pawseyenv
export MODULEPATH=${MODULEPATH}:${LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX}
module load pawsey
module load python/3.11.6
module load py-pip/23.1.2-py3.11.6

python -m venv --system-site-packages .partial_qec_venv # We want to use the system site packages here for integration with the Setonix systems
source .partial_qec_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch torchvision pennylane pennylane-lightning

# The user may have to manually install the following packages since they are not on the quantum parition of Setonix.
python -m pip install scipy rustworkx typing_extensions 

