"""
This module contains several noise models to pass into the HybridModel class (defined in qml_modules.py).
"""

import pennylane as qml
from superop import SuperOpTools
from scipy.linalg import expm, logm

def depolarising_single_qubit(p_depol: float, p_damping: float = 0) -> qml.NoiseModel:
    """
    Returns a depolarising noise model for RY gates with depolarising strength p.
    Also adds amplitude damping noise with strength amplitude_damping.
    """

    if p_damping == 0 and p_depol >= 0:
        return qml.NoiseModel({qml.noise.op_eq(qml.RY): qml.noise.partial_wires(qml.DepolarizingChannel, p = p_depol)})
    
    depol_channel_kraus_reps = qml.DepolarizingChannel.compute_kraus_matrices(p_depol)
    damping_channel_kraus_reps = qml.AmplitudeDamping.compute_kraus_matrices(p_damping)
    depol_channel_superop_reps, damping_channel_superop_reps = SuperOpTools.kraus2superop(depol_channel_kraus_reps), SuperOpTools.kraus2superop(damping_channel_kraus_reps)
    kraus_list = SuperOpTools.superop2kraus(expm(logm(depol_channel_superop_reps) + logm(damping_channel_superop_reps)))
    
    def depol_and_damping(op, **metadata):
        qml.QubitChannel(kraus_list, wires=op.wires)
    
    return qml.NoiseModel({qml.noise.op_eq(qml.RY): depol_and_damping})

def depolarising_two_qubit(p: float, num_qubits: int, phi: dict[str, float]|None = None) -> qml.NoiseModel:
    """
    Returns a depolarising noise model for CZ gates with depolarising strength p.
    Also adds crosstalk noise with strength phi (a dictionary of Pauli-Pauli interaction strengths).
    """

    theta = list(phi.values()) if phi is not None else [0.0] * 15
    unitary_channel_adjoint_rep = [qml.SpecialUnitary.compute_matrix(theta, num_wires=2)]

    def crosstalk_noise(op, **params):
        crosstalk_nearest_neighbour_wires_1 = sorted([(op.wires[0]-1) % num_qubits, op.wires[0]])
        crosstalk_nearest_neighbour_wires_2 = sorted([op.wires[1], (op.wires[1]+1) % num_qubits])
        qml.QubitChannel(unitary_channel_adjoint_rep, wires=crosstalk_nearest_neighbour_wires_1)
        qml.QubitChannel(unitary_channel_adjoint_rep, wires=crosstalk_nearest_neighbour_wires_2)
    
    depolarising_channel_kraus_reps = qml.DepolarizingChannel.compute_kraus_matrices(p)
    two_qubit_depolarising_channel_kraus_reps = [qml.math.kron(k1, k2) for k1 in depolarising_channel_kraus_reps for k2 in depolarising_channel_kraus_reps]

    def depolarising_noise(op, **params):
        qml.QubitChannel(two_qubit_depolarising_channel_kraus_reps, wires=op.wires)
    
    return qml.NoiseModel({
        qml.noise.op_eq(qml.CZ): crosstalk_noise,
        qml.noise.op_eq(qml.CZ): depolarising_noise
    })

