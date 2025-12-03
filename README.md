# 02456 Deep Learning - Project Group 101

## EMNIST Balanced Character Recognition with JAX

This repository contains our implementation of a flexible feedforward neural network built from scratch using JAX for the DTU course 02456 Deep Learning.

DTU Course 02456 Deep Learning project implementing a configurable JAX neural network for EMNIST Balanced character recognition. Includes hyperparameter optimization via Weights &amp; Biases sweeps and comprehensive performance analysis.


### Key Features
- Custom JAX-based neural network with configurable architecture
- Support for multiple activation functions (ReLU, Tanh, Sigmoid)
- Multiple optimizers (SGD, Adam, RMSprop)
- Regularization techniques (Dropout, L2 weight decay, Gradient clipping)
- Hyperparameter optimization using Weights & Biases Bayesian sweeps
- EMNIST Balanced dataset (47 character classes: digits + letters)
- Comprehensive evaluation with confusion matrix analysis

### Best Model Performance
- **Test Accuracy:** 87.12%
- **Architecture:** [512, 512, 512] with ReLU activation
- **Optimizer:** Adam with learning rate 0.358
- **Regularization:** Dropout (0.046), L2 (1e-8)

### Project Structure
- `JAXNet.py` - Core neural network implementation
- `JAXNet_E47B_Sweep.py` - Hyperparameter sweep script
- `JAXNet_Colab_Single_Run.ipynb` - Best model training and evaluation
- Results and analysis notebooks

**Course:** 02456 Deep Learning, DTU  
**Group:** 101  
**Semester:** Fall 2025
