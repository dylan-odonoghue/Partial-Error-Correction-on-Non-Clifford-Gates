import pennylane as qml
from pennylane import numpy as np
import torch
import torch.distributed as dist
from qvc_model import HybridModel
from noise_models import depolarising_single_qubit, depolarising_two_qubit
import pickle
from torchvision import transforms
import torchvision
import random

"""
Training loop for the QVC.

Implements the ADAM optimiser setup from Kang et al.:
  - Learning rate: 0.005
  - Batch size: 50 images
  - Loss: cross-entropy with softmax activation on Pauli-Z expectation values
  - Gradient: adjoint differentiation (via PennyLane)
  - Metrics logged: loss, avg gradient^2, test accuracy (every test_every batches)
"""

def preprocess_MNIST_data(num_qubits: int, train_size: int = 60000, test_size: int = 10000, random_seed: int = 42):
    """
    Preprocess the MNIST dataset for training and testing.
    Args:
        num_qubits: Number of qubits in the quantum circuit.
        train_size: Number of training samples to use (default is 60000).
        test_size: Number of testing samples to use (default is 10000).
        random_seed: Seed for random number generation.
    Returns:
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
    """
    if train_size > 60000:
        print("WARNING: train_size should be <= 60000. Using the full dataset instead.", flush=True)
    if test_size > 10000:
        print("WARNING: test_size should be <= 10000. Using the full dataset instead.", flush=True)

    im_size = round(np.sqrt(2**num_qubits)) # pyright: ignore[reportAttributeAccessIssue]

    # Resize images, convert to tensors
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
        transforms.Lambda(lambda x: x / torch.norm(x)),
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Randomly select a subset of the dataset based on train_size and test_size
    random.seed(random_seed)
    train_indices = random.sample(range(len(train_dataset)), min(train_size, len(train_dataset)))
    test_indices  = random.sample(range(len(test_dataset)),  min(test_size,  len(test_dataset)))

    train_dataset = torch.utils.data.Subset(train_dataset, indices=train_indices)
    test_dataset  = torch.utils.data.Subset(test_dataset,  indices=test_indices)

    return train_dataset, test_dataset



def serial_job(num_qubits: int, layers: int = 2, num_epochs: int = 5, batch_size: int = 50, noise_model=None, **kwargs):
    """
    Function to run the Hybrid Quantum-Classical model. This function is called by the main script.
    """
    rank             = kwargs.pop('rank', 0)  # Default rank is 0 for serial execution
    lr               = kwargs.pop('lr', 0.005)  # Learning rate for the optimizer
    num_shots        = kwargs.pop('num_shots', 10000)  # Number of shots for quantum measurements
    train_size       = kwargs.pop('train_size', 60000)  # Number of training samples
    test_size        = kwargs.pop('test_size', 10000)  # Number of testing samples
    name_extension   = kwargs.pop('name_extension', "")  # Optional name extension for logging
    noise_model_name = noise_model.metadata['name'] if noise_model is not None else "none"  # Name of the noise model for logging



    if name := kwargs.pop('name', None) is None:
        name = f"training_qubits_{num_qubits}_layers_{layers}_epochs_{num_epochs}_batch_{batch_size}_shots_{num_shots}_noise_{noise_model_name}{name_extension}"
    else:
        name = kwargs['name']
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True) if rank == 0 else None

    # Use DataLoader to create batches
    train_dataset, test_dataset = preprocess_MNIST_data(num_qubits, train_size=train_size, test_size=test_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if noise_model is not None:
        dev=qml.device("default.mixed", wires=num_qubits)
        print("Using default.mixed device.", flush=True) if rank == 0 else None
    else:
        try:
            dev=qml.device("lightning.gpu", wires=num_qubits)
            print("Using Lightning GPU device.", flush=True) if rank == 0 else None
        except:
            dev=qml.device("default.qubit", wires=num_qubits)
            print("Using default.qubit device.", flush=True) if rank == 0 else None
    weight_shapes = {"weights": (layers, num_qubits, 3)}  # 3 parameters represent the Euler angles for each qubit in the layer

    # Define model, loss function, and optimizer
    model     = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model=noise_model, **kwargs).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    results = {
        'training_samples': [0],
        'accuracies': {},
        'loss_values': [],
        'gradients': [],
    }
    trained_samples = 0
    
    model.train()
    for epoch in range(num_epochs):
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # Logging 
            trained_samples += len(inputs)
            results["training_samples"].append(trained_samples)
            results["loss_values"].append(loss.item())

            # Gradient logging: mean squared gradient over quantum params only
            quantum_grads = optimizer.param_groups[0]["params"][0].grad
            if quantum_grads is not None:
                mean_grad_sq = quantum_grads.pow(2).mean().item()
                results["gradients"].append(mean_grad_sq)
            else:
                results["gradients"].append(float("nan"))

            if trained_samples % (batch_size * 10) == 0:  # Log every 10 batches
                print(f"Samples seen {trained_samples:<6}",
                  f"loss {results['loss_values'][-1]:.4f} ",
                  f"grad² {results['gradients'][-1]:.2e} "
                  , sep="| ", flush=True) if rank == 0 else None

        # Test evaluation after each epoch
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)

        results["accuracies"][trained_samples] = correct / total
        model.train()
    if kwargs.get('save_results', True):
        with open(f"../results/{name}.pkl", "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

if __name__ == "__main__":
    # Example usage of the serial job function
    phi = {f"{P}{Q}": 0.0 for P in "IXYZ" for Q in "IXYZ" if not (P == "I" and Q == "I")}
    phi["ZZ"] = 0.00116
    noise_model = depolarising_two_qubit(p_depol=0.01, num_qubits=6, phi=phi)  
    noise_model = None
    serial_job(6, noise_model=noise_model, num_shots=10, batch_size=1, train_size=600, test_size=250, num_epochs=1, save_results=False)