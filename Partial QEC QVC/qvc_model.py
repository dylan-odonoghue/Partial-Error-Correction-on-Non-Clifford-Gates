"""
QVC (Quantum Variational Classifier) model.

Implements the circuit architecture from:
  Kang et al., "Almost fault-tolerant quantum machine learning
  with drastic overhead reduction" (arXiv:2507.18954)

Circuit structure (Fig. 3b):
  - Amplitude encoding of input data
  - L layers of:
      * Single-qubit rotations: R_Z(alpha) R_Y(beta) R_Z(gamma) per qubit
        (Euler angle decomposition, 3 trainable params per qubit per layer)
      * Fixed entangling layer: linear chain of CZ gates (no trainable params)
  - Measurement of Pauli-Z on each qubit
  - Softmax + cross-entropy loss (computed in training loop)

Unspecified details and the choices made here:
  - CZ entangling topology: linear chain (qubit i -- qubit i+1),
    the standard assumption for this circuit family.
  - Adjoint differentiation for gradients (exact, GPU-efficient).
"""

import pennylane as qml
import pennylane.numpy as pnp
import numpy as np


def make_device(n_qubits: int, n_shots, use_gpu: bool = True):
    """
    Create a PennyLane device.

    Args:
        n_qubits:  Number of qubits in the circuit.
        n_shots:   Number of measurement shots. None = exact expectation values.
                   Paper uses 10000 shots to simulate real hardware.
        use_gpu:   Use lightning.gpu (NVIDIA cuQuantum). Falls back to
                   lightning.qubit if GPU is unavailable.
    """
    if use_gpu:
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits, shots=n_shots)
            return dev
        except Exception:
            print("Warning: lightning.gpu unavailable, falling back to lightning.qubit.")
    return qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)


def amplitude_encode(x: np.ndarray, n_qubits: int) -> None:
    """
    Amplitude-encode a normalised input vector into the circuit.

    For n qubits, encodes a 2^n-dimensional vector into the quantum state.
    The paper uses this to encode 32x32 = 1024-pixel images on 10 qubits.

    x must be pre-normalised to unit norm.
    """
    qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=False, pad_with=0.0)


def entangling_layer(n_qubits: int) -> None:
    """
    Fixed linear-chain CZ entangling layer (no trainable parameters).
    Qubit connectivity: 0-1, 1-2, ..., (n-2)-(n-1).
    """
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i + 1])


def variational_layer(params_layer: pnp.ndarray, n_qubits: int) -> None:
    """
    One variational layer: Euler-angle single-qubit rotations on all qubits,
    followed by the CZ entangling layer.

    Args:
        params_layer: Shape (n_qubits, 3). The three Euler angles
                      [alpha, beta, gamma] per qubit. Gate applied is:
                      R_Z(alpha) R_Y(beta) R_Z(gamma)
        n_qubits:     Number of qubits.
    """
    for q in range(n_qubits):
        alpha, beta, gamma = params_layer[q]
        qml.RZ(gamma, wires=q)   # applied right-to-left: RZ(gamma) first
        qml.RY(beta,  wires=q)
        qml.RZ(alpha, wires=q)
    entangling_layer(n_qubits)


def build_qvc_circuit(dev, n_qubits: int, n_layers: int):
    """
    Build and return the QVC QNode.

    Returns a function qvc(x, params) -> array of shape (n_qubits,)
    containing <Z_j> for each qubit j.

    Args:
        dev:      PennyLane device.
        n_qubits: Number of qubits.
        n_layers: Number of variational layers L.
    """
    @qml.qnode(dev, diff_method="adjoint")
    def qvc(x: pnp.ndarray, params: pnp.ndarray) -> pnp.ndarray:
        """
        Args:
            x:      Input vector, shape (2**n_qubits,), pre-normalised.
            params: Variational parameters, shape (n_layers, n_qubits, 3).

        Returns:
            Array of shape (n_qubits,) with Pauli-Z expectation values.
        """
        amplitude_encode(x, n_qubits)
        for l in range(n_layers):
            variational_layer(params[l], n_qubits)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return qvc


def init_params(n_layers: int, n_qubits: int, seed: int = 42) -> pnp.ndarray:
    """
    Randomly initialise variational parameters in [-pi, pi].

    Shape: (n_layers, n_qubits, 3)
    """
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3))
    return pnp.array(vals, requires_grad=True)
