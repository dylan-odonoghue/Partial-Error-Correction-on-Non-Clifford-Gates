import os
import numpy as np
import noise_models
from train import serial_job, parallel_job
import argparse

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))  # Get the SLURM array task ID, default to 0 if not set

# Define noise configurations — one per array index
#p_values = [0] + list(np.logspace(-5, np.log10(5.11e-3), 9)) # Evenly spaced values from 1e-5 to 5.11e-3, plus a noise-free baseline at index 0
p_values = [0, 1.00e-5, 3.48e-5, 1.21e-4, 4.22e-4, 1.47e-3, 1.99e-3, 5.11e-3]  # Match the values used in the paper for consistency
p = p_values[task_id]

# Build noise model (None for noise-free baseline)
noise_model = None if p == 0 else noise_models.depolarising_single_qubit(p)

print(f"Task {task_id}: p = {p}", flush=True)

# Run
serial_job(
    num_qubits=6,
    layers=75,
    n_epochs=1,
    batch_size=50,
    test_size=250,
    noise_model=noise_model,
    num_shots=10000,
    divide_by=100,
    name=f"replicarun_6_qubits_{p:.2e}_depolarising_noise"
)