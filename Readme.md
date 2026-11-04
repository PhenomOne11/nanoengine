# nanoengine

### A From-Scratch Deep Learning Training Framework Built with PyTorch

A modular deep learning framework featuring custom neural network layers, optimizer implementation, and a ResNet-9 architecture for CIFAR-10 image classification. Built to explore the fundamentals of deep learning systems, model architecture design, and training optimization.

---

## Overview

**nanoengine** is a custom deep learning implementation developed using PyTorch, focusing on understanding and implementing core components of neural network training rather than relying entirely on high-level abstractions.

The project combines modular layer implementations, a custom AdamW optimizer, residual network architecture, and optimized training techniques to build and evaluate a CIFAR-10 image classifier.

## Key Features

* **Custom Layer Architecture:** Modular layer implementations with native PyTorch autograd parameter binding.
* **Custom AdamW Optimizer:** Implements decoupled weight decay and in-place moment buffer updates.
* **ResNet-9 Architecture:** A nine-layer residual convolutional neural network with adaptive pooling and multi-stage skip connections.
* **Learning Rate Scheduling:** Cosine Annealing learning rate schedule.
* **Data Augmentation:** Reflection padding and random horizontal flipping.
* **Performance Evaluation:** Validation accuracy, parameter count, inference latency, and batch throughput measurements.

## Architecture

The framework is organized into three primary components:

| Component | Implementation         | Description                                    |
| --------- | ---------------------- | ---------------------------------------------- |
| Layers    | `src/layers/`          | Custom modular neural network layers           |
| Optimizer | `src/optim/adamw.py`   | Custom AdamW optimization algorithm            |
| Model     | `src/models/resnet.py` | ResNet-9 architecture for image classification |

### 1. Custom Neural Network Layers

Implemented modular components with native autograd parameter binding.

* Base layer implementation.
* Convolutional layers.
* Fused convolution operations.

### 2. Custom AdamW Optimizer

A custom implementation of AdamW featuring:

* First-moment estimation.
* Second-moment estimation.
* Decoupled weight decay.
* In-place moment buffer updates.

### 3. ResNet-9

A nine-layer residual convolutional neural network featuring:

* Multi-stage residual connections.
* Convolutional feature extraction.
* Adaptive average pooling.
* CIFAR-10 classification output.

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

*Benchmark figures represent the recorded results from the current implementation and evaluation environment.*

## Project Structure

```text
nanoengine/
│
├── src/
│   ├── layers/
│   │   ├── base.py
│   │   └── conv.py
│   │
│   ├── models/
│   │   └── resnet.py
│   │
│   └── optim/
│       └── adamw.py
│
├── scripts/
│   └── train.py
│
├── .gitignore
└── README.md
```

## Technology Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Numerical Computing:** NumPy
* **Dataset:** CIFAR-10
* **Model Architecture:** ResNet-9
* **Optimization:** Custom AdamW
* **Learning Rate Scheduling:** Cosine Annealing

## Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* NumPy
* CUDA-compatible GPU (recommended for accelerated training)

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/nanoengine.git
cd nanoengine
```

Install the required dependencies:

```bash
pip install torch torchvision numpy
```

### Training

Train the ResNet-9 model using the following command:

```bash
python scripts/train.py --epochs 20 --batch-size 128 --lr 0.001 --device cuda
```

**Training configuration:**

* `--epochs 20`: Train for 20 epochs.
* `--batch-size 128`: Process 128 images per batch.
* `--lr 0.001`: Set the initial learning rate to 0.001.
* `--device cuda`: Run training on an NVIDIA CUDA-compatible GPU.

For CPU-based training, use:

```bash
python scripts/train.py --epochs 20 --batch-size 128 --lr 0.001 --device cpu
```

*Note: Adjust the training configuration and dataset settings in the training script if required.*

## Project Objectives

The primary objectives of nanoengine are:

* Understand the internal mechanics of deep learning training.
* Implement neural network components using PyTorch.
* Develop an optimizer from scratch.
* Design and train a residual CNN architecture.
* Evaluate model accuracy and computational performance.

## Future Improvements

* Expand the custom layer library.
* Implement additional optimization algorithms.
* Introduce mixed-precision training.
* Explore distributed training.
* Improve inference efficiency.
* Add automated testing and benchmarking.

---

**nanoengine** — Exploring the fundamentals of deep learning systems through implementation.
