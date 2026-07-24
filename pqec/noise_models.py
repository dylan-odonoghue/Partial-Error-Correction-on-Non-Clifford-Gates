"""
This module contains several noise models to pass into the HybridModel class (defined in qvc_model.py).
"""

from dataclasses import dataclass
import pennylane as qml
import numpy as np
from .superop import SuperOpTools
from scipy.linalg import expm, logm

def depolarising_single_qubit(p_depol: float, p_damping: float = 0) -> qml.NoiseModel:
    """
    Returns a depolarising noise model for RY gates with depolarising strength p_depol.

    Also adds amplitude damping noise with strength p_damping.
    """
    assert 0 <= p_depol <= 1, "Depolarising probability must be between 0 and 1."
    assert 0 <= p_damping <= 1, "Amplitude damping probability must be between 0 and 1."

    if p_damping == 0 and p_depol >= 0:
        return qml.NoiseModel({qml.noise.op_eq(qml.RY): qml.noise.partial_wires(qml.DepolarizingChannel, p_depol)}, 
                              name = f"single-qubit-depol(p-depol={p_depol})",
                              depolarising_noise = p_depol)
    
    depol_channel_kraus_reps = qml.DepolarizingChannel.compute_kraus_matrices(p_depol)
    damping_channel_kraus_reps = qml.AmplitudeDamping.compute_kraus_matrices(p_damping)
    depol_channel_superop_reps, damping_channel_superop_reps = SuperOpTools.kraus2superop(depol_channel_kraus_reps), SuperOpTools.kraus2superop(damping_channel_kraus_reps)
    kraus_list = SuperOpTools.superop2kraus(expm(logm(depol_channel_superop_reps) + logm(damping_channel_superop_reps)))
    
    def depol_and_damping(op, **metadata):
        qml.QubitChannel(kraus_list, wires=op.wires)
    
    return qml.NoiseModel({qml.noise.op_eq(qml.RY): depol_and_damping}, 
                          name=f"single-qubit-depol-damp(p-depol={p_depol}, p-damp={p_damping})",
                          depolarising_noise=p_depol, damping_noise=p_damping)

def depolarising_two_qubit(p_depol: float, num_qubits: int, phi: dict[str, float]|None = None) -> qml.NoiseModel:
    """
    Returns a depolarising noise model for CZ gates with depolarising strength p_depol.

    Also adds crosstalk noise with strength phi (a dictionary of Pauli-Pauli interaction strengths).
    """
    assert 0 <= p_depol <= 1, "Depolarising probability must be between 0 and 1."

    # Call phi to construct theta
    if phi is not None:
        theta = [phi.get(f"{P}{Q}", 0.0) for P in "IXYZ" for Q in "IXYZ" if not (P == "I" and Q == "I")]
        assert all(-np.pi <= val <= np.pi for val in theta), "All values in phi must be between -pi and pi."
    else:
        theta = [0.0] * 15
    

    unitary_crosstalk_channel = [qml.SpecialUnitary.compute_matrix(theta, num_wires=2)]

    def crosstalk_noise(op, **params):
        crosstalk_nearest_neighbour_wires_1 = sorted([(op.wires[0]-1) % num_qubits, op.wires[0]])
        crosstalk_nearest_neighbour_wires_2 = sorted([op.wires[1], (op.wires[1]+1) % num_qubits])
        qml.QubitChannel(unitary_crosstalk_channel, wires=crosstalk_nearest_neighbour_wires_1)
        qml.QubitChannel(unitary_crosstalk_channel, wires=crosstalk_nearest_neighbour_wires_2)
    
    depolarising_channel_kraus_reps = qml.DepolarizingChannel.compute_kraus_matrices(p_depol)
    two_qubit_depolarising_channel_kraus_reps = [qml.math.kron(k1, k2) for k1 in depolarising_channel_kraus_reps for k2 in depolarising_channel_kraus_reps] # pyright: ignore[reportAttributeAccessIssue]

    def depolarising_noise(op, **params):
        qml.QubitChannel(two_qubit_depolarising_channel_kraus_reps, wires=op.wires)
    
    if phi is None or all(val == 0 for val in phi.values()):
        name = f"two-qubit-depol(p-depol={p_depol})"
    else:
        name = f"two-qubit-depol-crosstalk(p-depol={p_depol})"

    return qml.NoiseModel({
        qml.noise.op_eq(qml.CZ): crosstalk_noise,
        qml.noise.op_eq(qml.CZ): depolarising_noise
    }, name=name, depolarising_noise=p_depol, crosstalk_noise=phi)
