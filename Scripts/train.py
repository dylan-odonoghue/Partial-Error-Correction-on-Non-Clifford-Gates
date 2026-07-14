import pennylane as qml
from pennylane import numpy as np
import torch
import torch.distributed as dist
from qvc_model import HybridModel
from noise_models import depolarising_single_qubit, depolarising_two_qubit
import pickle
from torchvision import transforms
import torchvision

"""
Training loop for the QVC.

Implements the ADAM optimiser setup from Kang et al.:
  - Learning rate: 0.005
  - Batch size: 50 images
  - Loss: cross-entropy with softmax activation on Pauli-Z expectation values
  - Gradient: adjoint differentiation (via PennyLane)
  - Metrics logged: loss, avg gradient^2, test accuracy (every test_every batches)
"""

def preprocess_MNIST_data(num_qubits: int, batch_size: int = 50, divide_by: int = 10):
    """
    Preprocess the MNIST dataset for training and testing.
    Args:
        num_qubits: Number of qubits in the quantum circuit.
        batch_size: Batch size for training.
        divide_by: Value to divide the dataset size by.
    Returns:
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
    """
    im_size = int(np.sqrt(2**num_qubits)) # pyright: ignore[reportAttributeAccessIssue]

    # Resize images, convert to tensors
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
        transforms.Lambda(lambda x: x / torch.norm(x)),
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), divide_by))
    test_dataset = torch.utils.data.Subset(test_dataset, indices=range(0, len(test_dataset), divide_by))

    return train_dataset, test_dataset



def serial_job(num_qubits, layers=2, n_epochs=5, batch_size=50, noise_model=None, **kwargs):
    """
    Function to run the Hybrid Quantum-Classical model. This function is called by the main script.
    """
    divide_by = 1 if 'divide_by' not in kwargs else kwargs['divide_by']  # Use a smaller subset of the dataset for quicker training
    rank = 0 if 'rank' not in kwargs else kwargs['rank']  # Default rank is 0 for serial execution
    lr = 0.005 if 'lr' not in kwargs else kwargs['lr']  # Learning rate for the optimizer
    num_shots = 10000 if 'num_shots' not in kwargs else kwargs['num_shots']  # Number of shots for quantum measurements

    if name := kwargs.get('name') is None:
        name = f"unnamed_qubits_{num_qubits}_layers_{layers}_epochs_{n_epochs}_batch_{batch_size}_shots_{num_shots}"
    else:
        name = kwargs['name']
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True) if rank == 0 else None

    # Use DataLoader to create batches
    train_dataset, test_dataset = preprocess_MNIST_data(num_qubits, batch_size=batch_size, divide_by=divide_by)
    # Use DataLoader to create batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    weight_shapes = {"weights": (layers, num_qubits, 3)}  # 3 parameters per qubit for RZ, RY, RZ rotations

    # Define model, loss function, and optimizer
    #phi = {f"{P}{Q}": 0.0 for P in "IXYZ" for Q in "IXYZ" if not (P == "I" and Q == "I")}
    #phi["ZZ"] = 0.00116
    model = HybridModel(dev=dev, device=device, num_qubits=num_qubits, weight_shapes=weight_shapes, noise_model=noise_model, num_shots=num_shots, **kwargs).to(device)
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
    for epoch in range(n_epochs):
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # NaN guard 
            if torch.isnan(loss):
                print(f"NaN loss encountered at {trained_samples} samples, skipping batch", flush=True) if rank == 0 else None
                continue

            # Backward pass
            loss.backward()
            optimizer.step()

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

            #print(f"  samples {trained_samples:>6} | loss {loss.item():.4f} | "
            #      f"grad² {results['gradients'][-1]:.2e}")

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

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"acc {results['accuracies'][trained_samples]:.2%} | "
              f"samples seen {trained_samples}", flush=True) if rank == 0 else None
    with open(f"../results/{name}.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

if __name__ == "__main__":
    # Example usage of the serial job function
    noise_model = None
    serial_job(6, noise_model=noise_model, num_shots=10, divide_by=100)  # Example: run the job with 6 qubits and 10 shots