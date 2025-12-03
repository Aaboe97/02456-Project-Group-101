# 02456 Deep Learning - Project Group 101

## EMNIST Balanced Character Recognition with JAX

Deep learning project implementing configurable feedforward neural networks from scratch for EMNIST Balanced character recognition (47 classes: digits + letters). Features comprehensive hyperparameter optimisation and performance analysis.

---

## 🚀 Quick Start

**To reproduce our best model (86.86% test accuracy):**

1. Open `JAXNet_Colab_Single_Run.ipynb` in Google Colab or VS Code with Colab extension
2. Run all cells to train the model and generate:
   - Training curves (loss & accuracy)
   - Confusion matrix
   - Misclassification analysis

**That's it!** This notebook contains our final optimized model with all hyperparameters pre-configured.

**Note:** If you want to generate training plots beyond the confusion matrix and misclassification analysis, set `use_wandb=True` in the notebook and provide your WandB API key when prompted. Familiarity with [Weights & Biases](https://wandb.ai) is expected for advanced logging and visualisation features.


---

## 📁 Repository Structure

### Core Implementations

| File | Description |
|------|-------------|
| **`JAXNet.py`** | Core neural network module with JAX (base class, training loop, evaluation) |
| **`PyNet.py`** | Core neural network module with NumPy (alternative implementation) |

### EMNIST Balanced (47 classes) - JAX Implementation

| File | Description |
|------|-------------|
| **`JAXNet_E47B.py`** | EMNIST Balanced standalone script (single run) |
| **`JAXNet_E47B_Sweep.py`** | EMNIST Balanced hyperparameter sweep script |
| **`JAXNet_Colab_Runner.ipynb`** | Notebook for running EMNIST sweeps in Colab |
| **`JAXNet_Colab_Single_Run.ipynb`** | ⭐ **Best model training + evaluation + confusion matrix** |

### EMNIST Balanced (47 classes) - NumPy Implementation

| File | Description |
|------|-------------|
| **`PyNet_E47B.py`** | EMNIST Balanced standalone script (single run) |
| **`PyNet_E47B_sweep.py`** | EMNIST Balanced hyperparameter sweep script |
| **`PyNet_Colab_Runner.ipynb`** | Notebook for running EMNIST sweeps in Colab |

### MNIST (10 classes) - Comparison Implementations

| File | Description |
|------|-------------|
| **`JAXNet_M10.py`** | MNIST standalone script (JAX) |
| **`JAXNet_M10_Sweep.py`** | MNIST hyperparameter sweep (JAX) |
| **`PyNet_M10.py`** | MNIST standalone script (NumPy) |
| **`PyNet_M10_Sweep.py`** | MNIST hyperparameter sweep (NumPy) |

### Configuration

| File | Description |
|------|-------------|
| **`sweep_config.yaml`** | WandB sweep configuration (Bayesian optimization) |

---

## 🎯 Best Model Performance

Achieved through Bayesian hyperparameter optimization (36 runs):

- **Test Accuracy:** 86.86%
- **Architecture:** 3 hidden layers [512, 512, 512]
- **Activation:** ReLU
- **Weight Init:** He initialization
- **Optimizer:** Adam (lr=0.358)
- **Regularization:** Dropout (p=0.046), L2 (λ=1e-8)
- **Batch Size:** 512
- **Training:** 100 epochs

---

## 🔧 Setup & Usage

### Prerequisites

```bash
pip install jax jaxlib numpy tensorflow-datasets wandb scikit-learn matplotlib
