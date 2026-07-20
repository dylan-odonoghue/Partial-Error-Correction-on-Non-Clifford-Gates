import os
import numpy as np
import noise_models
from train import serial_job
import argparse



parser = argparse.ArgumentParser(description="Run quantum neural network training with specified noise model.")
parser.add_argument("-n", "--num_qubits", type=int, default=6, help="Number of qubits in the quantum circuit.")
parser.add_argument("-l", "--layers", type=int, default=10, help="Number of layers in the quantum circuit.")
parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of training epochs.")
parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for training.")
parser.add_argument("-p", "--p", type=float, default = 0, help="Depolarising noise probability.")
parser.add_argument("-t", "--train_size", type=int, default=600, help="Number of training samples to use (default is 600).")
parser.add_argument("-s", "--test_size", type=int, default=250, help="Number of testing samples to use (default is 250).")
parser.add_argument("-a", "--array", action="store_true", help="Boolean flag to indicate if the script is running as part of a SLURM array job.")

args = parser.parse_args()

# Define noise configurations — one per array index
if args.array:
    if args.p != 0: 
        raise ValueError("When running as part of a SLURM array job, the noise probability 'p' should not be set manually. It will be determined by the SLURM_ARRAY_TASK_ID.")
    # Define noise configurations for array jobs
    noise_configs = [
        {"p": 0},  # Noise-free baseline
        {"p": 1.00e-5},
        {"p": 3.48e-5},
        {"p": 1.21e-4},
        {"p": 4.22e-4},
        {"p": 1.47e-3},
        {"p": 1.99e-3},
        {"p": 5.11e-3}
    ] # Matches the values used in Kang et al. for consistency

    # Get the SLURM array task ID, default to 0 if not set
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))  
    # Ensure task_id is within the bounds of noise_configs
    if task_id < len(noise_configs):
        args.p = noise_configs[task_id]["p"] # Set the noise probability based on the SLURM array task ID
        #name_extension = f""  # Name extension for logging
    else:
        raise ValueError(f"SLURM_ARRAY_TASK_ID {task_id} is out of bounds for the defined noise configurations.")
    

# Build noise model (None for noise-free baseline)
# If --array is set, the noise probability 'p' will be determined by the SLURM_ARRAY_TASK_ID, otherwise it will use the value provided by the user.
noise_model = None if args.p == 0 else noise_models.depolarising_single_qubit(args.p)

print(f"p = {args.p}", flush=True)

# Run the main training function with the specified parameters
serial_job(
    num_qubits=args.num_qubits,
    layers=args.layers,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    noise_model=noise_model,
    num_shots=10000,
    train_size=args.train_size,
    test_size=args.test_size,
    #name_extension=name_extension if args.array else None
)