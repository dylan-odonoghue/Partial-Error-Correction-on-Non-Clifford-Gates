"""
Plot training results — reproduces the style of Fig. 4a/4c from Kang et al.

Usage:
    python plot_results.py --results_dir results --run_name qvc75_noisefree
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_history(results_dir: str, run_name: str) -> dict:
    path = os.path.join(results_dir, f"{run_name}_history.json")
    with open(path) as f:
        return json.load(f)


def plot(history: dict, run_name: str, save_path: str | None = None):
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    samples = np.array(history["samples_seen"])
    test_samples = np.array(history["test_samples_seen"])

    # ── Left: test accuracy ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(test_samples, history["test_accuracy"],
             color="black", lw=1.5, label=run_name)
    ax1.set_xlabel("Training sample size")
    ax1.set_ylabel("Classification success probability")
    ax1.set_ylim(0.0, 0.85)
    ax1.set_xlim(0, samples[-1])
    ax1.legend(fontsize=8)
    ax1.set_title("Test accuracy (noise-free baseline)")
    ax1.grid(alpha=0.3)

    # ── Right: loss + avg gradient² ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2_r = ax2.twinx()

    loss_vals = np.array(history["train_loss"])
    grad_sq   = np.array(history["avg_grad_sq"])

    l1, = ax2.plot(samples, loss_vals, color="steelblue", lw=1.0, label="Loss")
    l2, = ax2_r.semilogy(samples, grad_sq, color="tomato", lw=1.0,
                          label=r"$\mathbb{E}_k[(\nabla_{\theta_k}\mathcal{L})^2]$")

    ax2.set_xlabel("Training sample size")
    ax2.set_ylabel("Loss $\\mathcal{L}_{\\theta}$", color="steelblue")
    ax2_r.set_ylabel("Avg. gradient²", color="tomato")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2_r.tick_params(axis="y", labelcolor="tomato")
    ax2.set_xlim(0, samples[-1])
    ax2.set_title("Loss and gradient magnitude")

    lines = [l1, l2]
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(f"QVC noise-free baseline — {run_name}", fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--run_name",    default="qvc75_noisefree")
    p.add_argument("--save",        type=str, default=None,
                   help="Path to save figure (e.g. results/fig.png). "
                        "If not set, display interactively.")
    args = p.parse_args()

    history = load_history(args.results_dir, args.run_name)
    save_path = args.save or os.path.join(
        args.results_dir, f"{args.run_name}_training_curves.png"
    )
    plot(history, args.run_name, save_path=save_path)


if __name__ == "__main__":
    main()
