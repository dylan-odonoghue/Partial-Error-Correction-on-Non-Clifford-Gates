# Partial Error Correction on Non-Clifford Gates
This project extends work in partial error correction and analyses the effect of noise on the performance and trainability of quantum machine learning models. A noisy variational quantum circuit is simulated using Pennylane's `default.mixed` device, and PyTorch's ADAM optimiser is used to optimise the circuit. The results are saved into Pickle files and analysed in Jupyter Notebook.

## File Structure
```
├── README.md
├── debug.ipynb
├── main.py
├── pqec
│   ├── __init__.py
│   ├── noise_models.py
│   ├── qvc_model.py
│   ├── superop.py
│   └── train.py
├── result_analysis.ipynb
└── results
```

## Paper Abstract
Errors in current quantum computers significantly limit the practicality and scalability of quantum algorithms, as noise accumulates in deep circuits and degrades quantum information. Although quantum error correction (QEC) offers a path toward fault-tolerant computation, the substantial spacetime overhead of existing protocols limits their feasibility on near-term devices. Building on recent work in partial QEC for quantum machine learning (QML), where error correction is applied selectively to entangling two-qubit gates, this project investigates an alternative strategy in which QEC is instead targeted toward non-Clifford operations. This direction is motivated by the high overhead typically associated with fault-tolerant non-Clifford gate implementations. We test whether selectively protecting non-Clifford resources can improve QML accuracy and trainability while reducing overhead relative to more uniform error-correction schemes. Using representative QML models and realistic noise channels, including depolarising and decoherence-based noise, we evaluate the resulting performance-resource trade-offs and aim to identify which gate families most strongly contribute to noise-induced degradation in QML.

## Acknowledgements
This research was supported by the Commonwealth through an Australian Government Research Training Program Scholarship [DOI: https://doi.org/10.82133/C42F-K220].