#!/bin/bash
#SBATCH --job-name=qvc_noisefree
#SBATCH --account=pawsey0309
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qvc_noisefree_%j.out
#SBATCH --error=logs/qvc_noisefree_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
# Activate micromamba environment (adjust env name as needed)
export MAMBA_EXE="$MYSCRATCH/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="$MYSCRATCH/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate qvc   # <-- change to your env name

# Confirm GPU is visible
echo "=== GPU info ==="
nvidia-smi
echo "================"

# Move to project directory
cd $SLURM_SUBMIT_DIR

# ── Run ───────────────────────────────────────────────────────────────────────
python main.py \
    --n_qubits   10   \
    --n_layers   75   \
    --n_train    15000 \
    --n_test     250  \
    --batch_size 50   \
    --lr         0.005 \
    --n_shots    10000 \
    --n_epochs   1    \
    --seed       42   \
    --results_dir results \
    --run_name   qvc75_noisefree \
    --test_every 10   \
    --save_every 50   \
    --data_cache data/mnist_preprocessed.npz

echo "Job finished at $(date)"
