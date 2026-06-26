"""
Main entrypoint: train a noise-free QVC on MNIST.

Reproduces the p=0 baseline from Kang et al. (arXiv:2507.18954), Fig. 4a.

Usage (local / interactive):
    python main.py

Usage (Setonix, via SLURM):
    sbatch job.slurm

Key hyperparameters (matching the paper):
    n_qubits  = 10   (encodes 32x32 = 1024 pixels)
    n_layers  = 75   (QVC75 model)
    n_train   = 15000
    n_test    = 250
    batch_size= 50
    lr        = 0.005
    n_shots   = 10000  (simulates hardware shot noise; set None for exact)
"""

import argparse
import numpy as np

from qvc_model import make_device, build_qvc_circuit, init_params
from data_utils import load_and_preprocess_mnist
from train import train


def parse_args():
    p = argparse.ArgumentParser(description="Noise-free QVC training")
    p.add_argument("--n_qubits",   type=int,   default=10)
    p.add_argument("--n_layers",   type=int,   default=75,
                   help="Number of variational layers (paper: 50, 75, 100)")
    p.add_argument("--n_train",    type=int,   default=15000)
    p.add_argument("--n_test",     type=int,   default=250)
    p.add_argument("--batch_size", type=int,   default=50)
    p.add_argument("--lr",         type=float, default=0.005)
    p.add_argument("--n_epochs",   type=int,   default=1,
                   help="Passes through training set. 1 epoch = 300 batches "
                        "of 50 = 15000 samples, matching the paper's x-axis.")
    p.add_argument("--n_shots",    type=int,   default=10000,
                   help="Measurement shots. Set 0 for exact expectation values.")
    p.add_argument("--no_gpu",     action="store_true",
                   help="Force CPU (lightning.qubit) even if GPU is available.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--results_dir",type=str,   default="results")
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--test_every", type=int,   default=10)
    p.add_argument("--save_every", type=int,   default=50)
    p.add_argument("--data_cache", type=str,   default="data/mnist_preprocessed.npz")
    return p.parse_args()


def main():
    args = parse_args()

    n_shots = args.n_shots if args.n_shots > 0 else None
    use_gpu = not args.no_gpu

    run_name = args.run_name or (
        f"qvc{args.n_layers}_noisefree_"
        f"{'shots' + str(n_shots) if n_shots else 'exact'}"
    )

    print("=" * 60)
    print("QVC Noise-Free Baseline")
    print(f"  n_qubits  : {args.n_qubits}")
    print(f"  n_layers  : {args.n_layers}   (QVC{args.n_layers})")
    print(f"  n_train   : {args.n_train}")
    print(f"  n_test    : {args.n_test}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr        : {args.lr}")
    print(f"  n_shots   : {n_shots}")
    print(f"  use_gpu   : {use_gpu}")
    print(f"  seed      : {args.seed}")
    print(f"  run_name  : {run_name}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        cache_path=args.data_cache,
    )
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # Sanity check: amplitude encoding requires unit-norm inputs
    norms = np.linalg.norm(X_train, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Inputs are not unit-normalised!"

    # ── Device and circuit ────────────────────────────────────────────────────
    print(f"\nBuilding device (lightning.{'gpu' if use_gpu else 'qubit'})...")
    dev = make_device(args.n_qubits, n_shots=n_shots, use_gpu=use_gpu)
    qvc = build_qvc_circuit(dev, args.n_qubits, args.n_layers)
    print(f"  Circuit: {args.n_qubits} qubits x {args.n_layers} layers "
          f"= {args.n_layers * args.n_qubits * 3} trainable parameters")

    # ── Parameters ───────────────────────────────────────────────────────────
    params = init_params(args.n_layers, args.n_qubits, seed=args.seed)
    print(f"  Parameters initialised: shape {params.shape}")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    history = train(
        qvc=qvc,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_every=args.test_every,
        save_every=args.save_every,
        results_dir=args.results_dir,
        run_name=run_name,
        seed=args.seed,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    final_acc = history["test_accuracy"][-1] if history["test_accuracy"] else float("nan")
    print("\n" + "=" * 60)
    print(f"Training complete.")
    print(f"  Final test accuracy: {final_acc:.4f}")
    print(f"  Results saved to:    {args.results_dir}/{run_name}_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
