#!/bin/bash -l
#SBATCH --job-name=partial_qec
#SBATCH --account=pawsey1116
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Load necessary modules
module load pytorch/2.7.1-rocm6.3.3
export TMPDIR=/tmp/${USER}-${SLURM_JOB_ID}
mkdir -p $TMPDIR results logs

export NEWSCRATCH=/scratch/pawsey1116/dodonoghue/Partial-Error-Correction-on-Non-Clifford-Gates
export MYENV=.partial_qec_venv
export VENV_PATH=${NEWSCRATCH}/${MYENV}

# Move to project directory
cd $NEWSCRATCH

# Define the script to run
export PYTHON_SCRIPT_DIR=${NEWSCRATCH}/Scripts
export PYTHONSCRIPT=${PYTHON_SCRIPT_DIR}/main.py

#----
# Additional Settings needed for this Pytorch script:
export MIOPEN_USER_DB_PATH="${TMPDIR}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${TMPDIR}/miopen_cache"
mkdir -p $MIOPEN_USER_DB_PATH $MIOPEN_CUSTOM_CACHE_DIR
export RDZV_PORT=29500
export RDZV_HOST=$(hostname)

# Set MPI related environment variables. (Not all need to be set)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1

echo -e "\n\n#------------------------#"
echo "Code execution:"

srun -N ${SLURM_JOB_NUM_NODES} -n ${SLURM_NTASKS} -c ${OMP_NUM_THREADS} -m block:block:block \
 singularity exec ${SINGULARITY_CONTAINER} \
 bash -c "source ${VENV_PATH}/bin/activate && \
 python3 ${PYTHON_SCRIPT}"

#----
# Remove any potentially leftover /tmp directories from this job
rm -rf ${TMPDIR}
#----
#Printing information of finished job steps:
echo -e "\n\n#------------------------#"
echo "Printing information of finished jobs steps using sacct:"
sacct -j ${SLURM_JOBID} -o jobid%20,Start%20,elapsed%20
#----
#Done
echo -e "\n\n#------------------------#"
echo "Done"