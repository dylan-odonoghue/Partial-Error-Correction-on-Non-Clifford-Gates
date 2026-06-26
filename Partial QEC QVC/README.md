# Partial QEC QVC — Reproduction & Extension

Codebase for reproducing and extending Kang et al.,
*"Almost fault-tolerant quantum machine learning with drastic overhead reduction"*
(arXiv:2507.18954).

## File structure

```
partial_qec_qvc/
├── qvc_model.py      # Circuit architecture (QNode, amplitude encoding, layers)
├── data_utils.py     # MNIST loading, 32x32 resize, L2 normalisation, batching
├── train.py          # ADAM training loop, loss, checkpointing
├── main.py           # Entrypoint (CLI args)
├── plot_results.py   # Reproduce Fig. 4a/4c style plots
├── setup_env.sh      # One-time micromamba environment setup on Setonix
├── job.slurm         # SLURM job script for Setonix gpu partition
├── data/             # Cached preprocessed MNIST (created on first run)
├── results/          # Checkpoints and history JSON
└── logs/             # SLURM stdout/stderr
```

## Quickstart (Setonix)

```bash
# 1. Set up environment (once)
bash setup_env.sh

# 2. Submit job
mkdir -p logs results
sbatch job.slurm

# 3. Monitor
squeue -u dodonoghue
tail -f logs/qvc_noisefree_<JOBID>.out

# 4. Plot results (after training)
python plot_results.py --run_name qvc75_noisefree
```

## Circuit architecture

Matches Fig. 3b of the paper exactly:

- **Encoding**: amplitude encoding, 10 qubits → 1024-pixel (32×32) images
- **Layer**: `R_Z(α) R_Y(β) R_Z(γ)` on each qubit, then linear-chain CZ gates
- **Parameters**: `n_layers × n_qubits × 3` Euler angles, init uniform in `[-π, π]`
- **Measurement**: Pauli-Z expectation value on each qubit
- **Loss**: cross-entropy with softmax (Eq. 6)
- **Optimiser**: ADAM, lr=0.005, batch size=50

**Unspecified details and choices made:**
- CZ topology: linear chain (0–1, 1–2, …, 8–9). Standard assumption for this
  circuit family; paper does not specify.
- Gradient method: adjoint differentiation (exact, O(n_params) GPU memory).

## Planned extensions

- [ ] Noise-free baseline (this file) — `p=0` curve in Fig. 4a
- [ ] Partial QEC: noisy single-qubit gates, error-corrected CZ (paper's scheme)
- [ ] **Inverted partial QEC**: noisy CZ gates, error-corrected single-qubit
      rotations — the novel contribution
- [ ] Phase damping and thermal damping channels (Fig. 6, 7)
- [ ] QVCC variant (QVC + classical fully-connected layer)
