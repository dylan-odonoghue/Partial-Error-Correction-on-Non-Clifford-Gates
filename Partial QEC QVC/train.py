"""
Training loop for the QVC.

Implements the ADAM optimiser setup from Kang et al.:
  - Learning rate: 0.005
  - Batch size: 50 images
  - Loss: cross-entropy with softmax activation on Pauli-Z expectation values
  - Gradient: adjoint differentiation (via PennyLane)
  - Metrics logged: loss, avg gradient^2, test accuracy (every test_every batches)

Cross-entropy loss (Eq. 6 in paper):
  L(y_hat, y) = sum_j y_j * log(softmax(z_j))
where z_j = <Z_j> and softmax is applied over qubits.

Since each sample has exactly one true label a:
  L = log(softmax(z)_a)   (only the correct class term survives)
"""

import numpy as np
import pennylane.numpy as pnp
import pennylane as qml
import json
import time
import os
from typing import Callable


# ── Loss and metric functions ────────────────────────────────────────────────

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1D array."""
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def cross_entropy(z: np.ndarray, label: int) -> float:
    """
    Cross-entropy loss for a single sample.
    z: raw Pauli-Z expectation values (n_qubits,)
    label: integer class index 0-9
    """
    probs = softmax(z)
    return -np.log(probs[label] + 1e-12)


def predict(z: np.ndarray) -> int:
    """Predicted class = argmax of softmax(z)."""
    return int(np.argmax(softmax(z)))


# ── Batch loss (used for gradient computation) ───────────────────────────────

def batch_loss(
    params: pnp.ndarray,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    qvc: Callable,
) -> pnp.ndarray:
    """
    Mean cross-entropy loss over a batch.
    Differentiable w.r.t. params via PennyLane's adjoint method.
    """
    total_loss = pnp.array(0.0)
    for x, label in zip(X_batch, y_batch):
        z = pnp.stack(qvc(x, params))          # (n_qubits,)
        # softmax in pennylane-numpy for differentiability
        z_shifted = z - pnp.max(z)
        exp_z = pnp.exp(z_shifted)
        probs = exp_z / pnp.sum(exp_z)
        total_loss = total_loss + (-pnp.log(probs[label] + 1e-12))
    return total_loss / len(X_batch)


# ── Training loop ────────────────────────────────────────────────────────────

def train(
    qvc: Callable,
    params: pnp.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 1,
    batch_size: int = 50,
    lr: float = 0.005,
    test_every: int = 10,       # evaluate test accuracy every N batches
    save_every: int = 50,       # checkpoint params every N batches
    results_dir: str = "results",
    run_name: str = "qvc_noisefree",
    seed: int = 42,
) -> dict:
    """
    Train the QVC with ADAM.

    Args:
        qvc:         QNode circuit function.
        params:      Initial parameters, shape (n_layers, n_qubits, 3).
        X_train:     Training data, (n_train, 2**n_qubits).
        y_train:     Training labels, (n_train,).
        X_test:      Test data, (n_test, 2**n_qubits).
        y_test:      Test labels, (n_test,).
        n_epochs:    Number of passes through the training set.
        batch_size:  Samples per gradient step.
        lr:          ADAM learning rate (paper: 0.005).
        test_every:  Evaluate on test set every this many batches.
        save_every:  Save checkpoint every this many batches.
        results_dir: Directory for logs and checkpoints.
        run_name:    Prefix for output files.
        seed:        RNG seed for batch shuffling.

    Returns:
        Dictionary of training history.
    """
    os.makedirs(results_dir, exist_ok=True)

    opt = qml.AdamOptimizer(stepsize=lr)
    grad_fn = qml.grad(batch_loss)

    history = {
        "train_loss":       [],   # per batch
        "avg_grad_sq":      [],   # per batch: mean of (d_loss/d_theta)^2
        "test_accuracy":    [],   # per test evaluation
        "samples_seen":     [],   # x-axis: cumulative training samples
        "test_samples_seen":[],
    }

    samples_seen = 0
    batch_num = 0
    t_start = time.time()

    for epoch in range(n_epochs):
        rng = np.random.default_rng(seed + epoch)
        idx = rng.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_idx = idx[start:start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # ── Gradient step ────────────────────────────────────────────────
            grads = grad_fn(params, X_batch, y_batch, qvc)
            params = opt.apply_grad(grads, params)

            # ── Logging ──────────────────────────────────────────────────────
            loss_val = float(batch_loss(params, X_batch, y_batch, qvc))
            avg_grad_sq = float(pnp.mean(grads ** 2))

            samples_seen += len(X_batch)
            history["train_loss"].append(loss_val)
            history["avg_grad_sq"].append(avg_grad_sq)
            history["samples_seen"].append(samples_seen)

            elapsed = time.time() - t_start
            print(
                f"[{elapsed:7.1f}s] "
                f"epoch {epoch+1}/{n_epochs}  "
                f"batch {batch_num+1}  "
                f"samples {samples_seen:>6}  "
                f"loss {loss_val:.4f}  "
                f"grad² {avg_grad_sq:.2e}"
            )

            # ── Test accuracy ─────────────────────────────────────────────────
            if batch_num % test_every == 0:
                acc = evaluate(qvc, params, X_test, y_test)
                history["test_accuracy"].append(acc)
                history["test_samples_seen"].append(samples_seen)
                print(f"  >> test accuracy: {acc:.4f}")

            # ── Checkpoint ───────────────────────────────────────────────────
            if batch_num % save_every == 0:
                _save_checkpoint(params, history, results_dir, run_name, batch_num)

            batch_num += 1

    # Final checkpoint
    _save_checkpoint(params, history, results_dir, run_name, batch_num, final=True)
    return history


def evaluate(
    qvc: Callable,
    params: pnp.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Classification accuracy on the test set."""
    correct = 0
    for x, label in zip(X_test, y_test):
        z = np.array(qvc(x, params))
        if predict(z) == int(label):
            correct += 1
    return correct / len(y_test)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _save_checkpoint(params, history, results_dir, run_name, batch_num, final=False):
    suffix = "final" if final else f"batch{batch_num:05d}"
    param_path = os.path.join(results_dir, f"{run_name}_params_{suffix}.npy")
    hist_path  = os.path.join(results_dir, f"{run_name}_history.json")
    np.save(param_path, np.array(params))
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [checkpoint saved: {param_path}]")
