#!/bin/bash -l
#SBATCH --job-name=partial_qec
#SBATCH --account=pawsey1116-q
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Load necessary modules
module purge
module load pawsey pawseytools pawseyenv PrgEnv-nvidia craype-arm-grace
module load gcc-native-mixed/12.3
module load pawseyenv
export MODULEPATH=$MODULEPATH:$LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX
module load pawsey



export NEWSCRATCH=/scratch/pawsey1116/dodonoghue/Partial-Error-Correction-on-Non-Clifford-Gates

# Activate the virtual environment
VENV_PATH=${NEWSCRATCH}/.partial_qec_venv
source ${VENV_PATH}/bin/activate

# Move to scripts directory
cd $NEWSCRATCH/Scripts

# Define the script to run
export PYTHON_SCRIPT_DIR=${NEWSCRATCH}/Scripts
export PYTHON_SCRIPT=${PYTHON_SCRIPT_DIR}/main.py

# Set MPI related environment variables. (Not all need to be set)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1

echo -e "\n\n#------------------------#"
echo "Code execution:"

srun -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_NTASKS} --gres=gpu:1 -m block:block:block \
 python ${PYTHON_SCRIPT}
#----
#Printing information of finished job steps:
echo -e "\n\n#------------------------#"
echo "Printing information of finished jobs steps using sacct:"
sacct -j ${SLURM_JOBID} -o jobid%20,Start%20,elapsed%20
#----
#Done
echo -e "\n\n#------------------------#"
echo "Done"