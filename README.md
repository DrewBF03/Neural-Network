# Neural-Network
MNIST Handwritten Digit Recognizer

## Overview
This project implements a **3-layer fully connected, feed-forward neural network** in Java. It demonstrates training and evaluation of the network using **stochastic gradient descent (SGD)** and **backpropagation**. The network is applied to both synthetic data and the real-world task of recognizing handwritten digits using the MNIST dataset.

---

## Features

### Tiny Neural Network
- A **3-layer neural network** with:
  - Input layer: 4 nodes
  - Hidden layer: 3 nodes
  - Output layer: 2 nodes
- Predefined weights and biases for verification.
- Manual training over **six epochs** using small input/output pairs.
- Outputs diagnostic values to verify correct implementation.

### MNIST Handwritten Digit Recognition
- A scalable neural network for classifying **handwritten digits (0-9)** using the MNIST dataset.
- Features include:
  - Randomized initialization of weights and biases.
  - **One-hot encoding** for labels.
  - Scaling pixel values to the range [0, 1].
  - Training with 60,000 MNIST examples using SGD (30 epochs, mini-batch size of 10).
  - Evaluation on 10,000 test examples.

---

## Functionality
1. Train the network using the MNIST dataset.
2. Load pre-trained weights from a file.
3. Display accuracy statistics for training and testing datasets.
4. Visualize results:
   - ASCII representations of misclassified digits.
   - Correct and incorrect predictions.
5. Save the network state for future use.

---

## Requirements
- MNIST dataset in CSV format (available [here](https://pjreddie.com/projects/mnist-in-csv/)).

---
