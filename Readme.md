# nanoengine

### A From-Scratch Deep Learning Training Framework Built with PyTorch

**nanoengine** is a modular deep learning framework built to explore and implement the core components of neural network training. It features custom neural network layers, optimization algorithms, training callbacks, tensor operations, and a ResNet-9 architecture for CIFAR-10 image classification.

---

## Overview

Unlike implementations that rely entirely on high-level PyTorch abstractions, nanoengine focuses on building and understanding the underlying components of a deep learning training pipeline.

The framework integrates custom layer implementations, optimizer algorithms, learning rate scheduling, training callbacks, and residual network architecture into a modular codebase.

## Key Features

* **Custom Neural Network Layers:** Modular implementations of convolutional, linear, activation, and normalization layers.
* **Custom Optimization Algorithms:** From-scratch AdamW and SGD implementations.
* **ResNet-9 Architecture:** Residual convolutional neural network with multi-stage skip connections and adaptive pooling.
* **Tensor Operations:** Custom tensor-related functionality.
* **Training Engine:** Structured training workflow with modular execution.
* **Callback System:** Training callbacks, execution management, and tracking.
* **Learning Rate Scheduling:** Cosine Annealing learning rate scheduling.
* **Data Augmentation:** Reflection padding and random horizontal flipping.
* **Testing Suite:** Component-level tests covering layers, optimizers, and sequential operations.
* **Performance Benchmarking:** Evaluation of validation accuracy, inference latency, and batch throughput.

---

## Architecture

The framework is organized into five major components:

| Component  | Directory        | Responsibility                                       |
| ---------- | ---------------- | ---------------------------------------------------- |
| Callbacks  | `src/callbacks/` | Training execution, callbacks, and tracking          |
| Engine     | `src/engine/`    | Tensor operations                                    |
| Layers     | `src/layers/`    | Neural network building blocks                       |
| Models     | `src/models/`    | Model architectures                                  |
| Optimizers | `src/optim/`     | Optimization algorithms and learning rate scheduling |

### 1. Neural Network Layers

Custom modular implementations covering:

* Base layer functionality.
* Convolutional layers.
* Linear layers.
* Activation functions.
* Normalization layers.
* Container and sequential operations.

### 2. Optimization Algorithms

Custom implementations of:

* **AdamW:** Adaptive moment estimation with decoupled weight decay.
* **SGD:** Stochastic Gradient Descent.
* **Learning Rate Scheduler:** Cosine Annealing.

### 3. ResNet-9 Architecture

A nine-layer residual convolutional neural network designed for CIFAR-10 classification.

Key architectural components:

* Convolutional feature extraction.
* Residual skip connections.
* Multi-stage feature learning.
* Adaptive average pooling.
* Classification output layer.

### 4. Training Engine and Callbacks

A modular training infrastructure featuring:

* Tensor operations.
* Callback execution.
* Training workflow management.
* Experiment and metric tracking.

---

## Benchmark Results

| Metric                       |          Result |
| ---------------------------- | --------------: |
| Validation Accuracy          |      **91.85%** |
| Training Epochs              |              20 |
| Trainable Parameters         |       6,575,370 |
| Approximate Parameter Memory |        25.08 MB |
| Single-Image Latency         |         3.49 ms |
| Batch Throughput             |     3,723.5 FPS |
| Benchmark Batch Size         |             128 |
| Hardware                     | NVIDIA Tesla T4 |

*Results represent measurements from the current implementation and evaluation environment. Performance may vary depending on hardware and benchmarking conditions.*

---

## Project Structure

```text
nanoengine/
│
├── src/
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── runner.py
│   │   └── tracking.py
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   └── tensor.py
│   │
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── base.py
│   │   ├── containers.py
│   │   ├── conv.py
│   │   ├── linear.py
│   │   └── norm.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── resnet.py
│   │
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── adamw.py
│   │   ├── lr_scheduler.py
│   │   └── sgd.py
│   │
│   └── __init__.py
│
├── scripts/
│   ├── benchmark.py
│   ├── download_cifar.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_activations.py
│   ├── test_conv.py
│   ├── test_linear.py
│   ├── test_norm.py
│   ├── test_optim.py
│   ├── test_sequential.py
│   └── test_stage5.py
│
├── weights/
│   └── .gitkeep
│
├── confusion_matrix.png
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Technology Stack

| Technology | Purpose                                     |
| ---------- | ------------------------------------------- |
| Python     | Core programming language                   |
| PyTorch    | Deep learning and automatic differentiation |
| NumPy      | Numerical computing                         |
| CIFAR-10   | Image classification dataset                |
| CUDA       | GPU-accelerated training                    |
| Pytest     | Automated testing                           |

---

## Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* NumPy
* CUDA-compatible GPU (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nanoengine.git
cd nanoengine
```

### 2. Install Dependencies

```bash
pip install -e .
```

*Dependencies are managed through `pyproject.toml`.*

### 3. Download the Dataset

```bash
python scripts/download_cifar.py
```

### 4. Train the Model

Run the training pipeline with the following configuration:

```bash
python scripts/train.py --epochs 20 --batch-size 128 --lr 0.001 --device cuda
```

**Training configuration:**

| Argument           | Description                |
| ------------------ | -------------------------- |
| `--epochs 20`      | Number of training epochs  |
| `--batch-size 128` | Images processed per batch |
| `--lr 0.001`       | Initial learning rate      |
| `--device cuda`    | GPU-based training         |

For CPU-based training:

```bash
python scripts/train.py --epochs 20 --batch-size 128 --lr 0.001 --device cpu
```

### 5. Run Benchmarks

```bash
python scripts/benchmark.py
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## Testing

The project includes a dedicated test suite covering:

* Activation functions.
* Convolutional layers.
* Linear layers.
* Normalization layers.
* Optimization algorithms.
* Sequential containers.
* Additional training-stage functionality.

Run the complete test suite:

```bash
pytest tests/ -v
```

---

## Future Development

Potential areas of further development include:

* Additional neural network architectures.
* More optimization algorithms.
* Mixed-precision training.
* Distributed training support.
* Advanced model profiling.
* Improved memory and computational efficiency.

---

## Project Philosophy

**Understand the internals. Build the components. Measure the results.**

nanoengine explores deep learning from the implementation level, bringing together neural network architecture, optimization, training infrastructure, and performance evaluation in one modular framework.


