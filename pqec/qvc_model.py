# Import required libraries
import pennylane as qml
import torch
import torch.nn as nn
from typing import Union

class ShotNoise(nn.Module):
    """
    A stochastic layer to simulate shot noise in quantum circuits.
    
    This layer adds Gaussian noise to the expectation values of quantum measurements to simulate the effect of finite sampling (shot noise).
    This layer is still applied at model evaluation time. This is because the quantum circuit is still being simulated with a finite number of shots, and thus the expectation values will inherently have some noise due to the limited number of samples.
    Args:
        num_shots: The number of shots (samples) used in the quantum measurement.
        device: The PyTorch device (CPU or GPU) on which the computations will be performed.
    
    """
    def __init__(self, num_shots, device):
        super().__init__()
        self.num_shots = num_shots
        self.device = device
    def forward(self, x):
        """Simulate shot noise by adding Gaussian noise to the expectation values."""
        p_positive_eigvals = (x+1)/2
        variance = 4*p_positive_eigvals*(1-p_positive_eigvals)/self.num_shots
        expval_with_shotnoise = x + variance**0.5*torch.randn_like(x).to(self.device)
        return expval_with_shotnoise.clamp(min=-1,max=1)

# Quantum neural network
class HybridModel(nn.Module):

    """
    A hybrid quantum-classical neural network model.
    This model uses a quantum circuit for feature extraction followed by a classical neural network
    for classification.
    The quantum circuit is defined using PennyLane, and the classical part is a simple feedforward 
    neural network.
    The quantum circuit uses amplitude encoding and trainable parameters for quantum gates.
    The model can be configured with various noise models to simulate realistic quantum operations.
    Args:
        dev: The PennyLane device to set the mode of circuit execution.
        device: The PyTorch device (CPU or GPU) for the classical simulation.
        num_qubits: The number of qubits in the quantum circuit.
        weight_shapes: A dictionary defining the shapes of the weights in the quantum circuit.
        **kwargs: Additional keyword arguments for configuring the quantum circuit and noise model.
    Attributes:
        dev: The PennyLane device for the quantum circuit.
        device: The PyTorch device for the classical part.
        num_qubits: The number of qubits in the quantum circuit.
        quantum_circuit: A Pennylane quantum node that represents the quantum circuit.
        quantum: A Pennylane TorchLayer that wraps the quantum circuit, integrates with PyTorch.
        classical: A PyTorch linear layer that serves as the classical part of the model.
    Methods:
        forward(x): Defines the forward pass of the model, processing input data through the 
        quantum circuit and classical neural network.
    """

    def __init__(
        self, dev, device: torch.device, num_qubits: int, weight_shapes: dict[str, tuple], noise_model: Union[qml.NoiseModel, None] = None, **kwargs):
        super().__init__()
        self.dev = dev
        self.device = device
        self.num_qubits = num_qubits
        self.weight_shapes = weight_shapes
        self.kwargs = kwargs
        self.shot_noise = ShotNoise(num_shots=kwargs.get('num_shots', 10000), device=device)
        self.noise_model = noise_model
        self.classical = nn.Linear(num_qubits, 10)

        assert num_qubits > 0, "Number of qubits must be a positive integer."
        assert num_qubits % 2 == 0, "Number of qubits must be even to avoid padding in AmplitudeEmbedding."

        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            """
            The quantum circuit that processes the input data.
            Args:
                inputs: The input data to be encoded into the quantum circuit.
                weights: The trainable weights for the quantum gates in the circuit.
            Returns:
                The expectation values of the Pauli-Z operator for each qubit.
            """
            # Amplitude encoding of the input data
            qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)

            num_layers = weights.shape[0]
            for layer in range(num_layers):
                for qubit in range(self.num_qubits):
                    qml.RZ(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                for qubit in range(0, self.num_qubits, 2):
                    qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])
                for qubit in range(1, self.num_qubits, 2):
                    qml.CZ(wires=[qubit, (qubit+1)%self.num_qubits])
            # Measure the expectation values of Pauli-Z on each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_qubits)]

        self.quantum_circuit = quantum_circuit
        quantum_circuit_to_wrap = qml.add_noise(quantum_circuit, self.noise_model) if self.noise_model is not None else quantum_circuit
        self.quantum = qml.qnn.TorchLayer(quantum_circuit_to_wrap, self.weight_shapes) # pyright: ignore[reportCallIssue]

    def forward(self, x):
        """
        Forward pass of the hybrid model.
        Args:
            x: The input data to be processed by the model.
        Returns:
            The output of the classical neural network after processing the quantum circuit's output.
        """
        # Pass the input through the quantum circuit
        quantum_output = self.quantum(x).to(self.device)
        
        # Apply shot noise
        output = self.shot_noise(quantum_output).to(self.device)
        
        # Pass the quantum output through the classical neural network
        output = self.classical(output).to(self.device)
        
        return output

