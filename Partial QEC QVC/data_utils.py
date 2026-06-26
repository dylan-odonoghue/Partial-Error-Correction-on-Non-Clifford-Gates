"""
MNIST data loading and preprocessing for the QVC.

Preprocessing pipeline:
  1. Download raw MNIST (via torchvision or sklearn, whichever is available).
  2. Resize images to 32x32 (paper uses 32x32 -> 1024 pixels -> 10 qubits).
  3. Flatten to a 1024-dim vector.
  4. L2-normalise each vector so amplitude encoding is valid.

Amplitude encoding requires ||x||_2 = 1 exactly.
"""

import numpy as np
from typing import Tuple


def _load_mnist_sklearn() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST via sklearn (70k samples, 28x28, values 0-255)."""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32)       # (70000, 784)
    y = mnist.target.astype(np.int64)        # (70000,)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def _resize_28_to_32(X: np.ndarray) -> np.ndarray:
    """
    Resize (N, 784) MNIST images from 28x28 to 32x32 using bilinear interpolation.
    Returns (N, 1024).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for image resizing: pip install Pillow")

    N = X.shape[0]
    out = np.zeros((N, 1024), dtype=np.float32)
    for i in range(N):
        img = Image.fromarray(X[i].reshape(28, 28)).resize((32, 32), Image.BILINEAR)
        out[i] = np.array(img, dtype=np.float32).flatten()
    return out


def _l2_normalise(X: np.ndarray) -> np.ndarray:
    """L2-normalise each row. Adds small epsilon to avoid division by zero."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def load_and_preprocess_mnist(
    n_train: int = 15000,
    n_test: int = 250,
    seed: int = 42,
    cache_path: str = "data/mnist_preprocessed.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, resize, normalise, and subsample MNIST.

    Matches paper setup:
      - 15000 training images, drawn and ordered randomly
      - 250 test images

    Args:
        n_train:    Number of training samples.
        n_test:     Number of test samples.
        seed:       RNG seed for shuffling.
        cache_path: Path to cache the preprocessed arrays (saves re-download time).

    Returns:
        X_train: (n_train, 1024) float32, L2-normalised
        y_train: (n_train,)      int64,   labels 0-9
        X_test:  (n_test, 1024)  float32, L2-normalised
        y_test:  (n_test,)       int64,   labels 0-9
    """
    import os

    if os.path.exists(cache_path):
        print(f"Loading preprocessed MNIST from cache: {cache_path}")
        data = np.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    print("Downloading and preprocessing MNIST...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = _load_mnist_sklearn()

    # Resize 28x28 -> 32x32
    print("Resizing images to 32x32...")
    X_train_32 = _resize_28_to_32(X_train_raw)
    X_test_32  = _resize_28_to_32(X_test_raw)

    # L2 normalise
    X_train_norm = _l2_normalise(X_train_32)
    X_test_norm  = _l2_normalise(X_test_32)

    # Subsample: paper draws randomly
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(X_train_norm), size=n_train, replace=False)
    test_idx  = rng.choice(len(X_test_norm),  size=n_test,  replace=False)

    X_train = X_train_norm[train_idx]
    y_train = y_train_raw[train_idx]
    X_test  = X_test_norm[test_idx]
    y_test  = y_test_raw[test_idx]

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print(f"Cached preprocessed data to {cache_path}")

    return X_train, y_train, X_test, y_test


def get_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 50,
    seed: int = 0,
):
    """
    Yield (X_batch, y_batch) tuples, shuffled each call.
    Paper uses batch size 50.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
