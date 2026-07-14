import os
import numpy as np
import noise_models
from train import serial_job
import argparse

#task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))  # Get the SLURM array task ID, default to 0 if not set

parser = argparse.ArgumentParser(description="Run quantum neural network training with specified noise model.")
parser.add_argument("-n", "--num_qubits", type=int, default=6, help="Number of qubits in the quantum circuit.")
parser.add_argument("-l", "--layers", type=int, default=10, help="Number of layers in the quantum circuit.")
parser.add_argument("-e", "--n_epochs", type=int, default=1, help="Number of training epochs.")
parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for training.")
parser.add_argument("-p", "--p", type=float, default = 0, help="Depolarising noise probability.")
parser.add_argument("-d", "--divide_by", type=int, default=100, help="Divide the size of the dataset by this value.")

args = parser.parse_args()

# Define noise configurations — one per array index
#p_values = [0] + list(np.logspace(-5, np.log10(5.11e-3), 9)) # Evenly spaced values from 1e-5 to 5.11e-3, plus a noise-free baseline at index 0
#p_values = [0, 1.00e-5, 3.48e-5, 1.21e-4, 4.22e-4, 1.47e-3, 1.99e-3, 5.11e-3]  # Match the values used in the paper for consistency
#p = p_values[task_id]

# Build noise model (None for noise-free baseline)
noise_model = None if args.p == 0 else noise_models.depolarising_single_qubit(args.p)

print(f"p = {args.p}", flush=True)

# Run
serial_job(
    num_qubits=args.num_qubits,
    layers=args.layers,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    noise_model=noise_model,
    num_shots=10000,
    divide_by=args.divide_by,
    name=f"replicarun_{args.num_qubits}_qubits_{args.p:.2e}_depolarising_noise"
)