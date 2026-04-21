# neural-network-with-learnable-gates
elf-Pruning Neural Network
# Overview

This project implements a self-pruning neural network that learns to remove unnecessary connections during training. Instead of applying pruning after training, the model integrates a learnable gating mechanism that dynamically suppresses less important weights.

The implementation is tested on the CIFAR-10 dataset using PyTorch.

# Key Idea

Each weight in the network is associated with a learnable gate parameter:

Gate values are constrained between 0 and 1 using a sigmoid function

Effective weight:

w' = w × sigmoid(gate_score)
If a gate approaches 0, the corresponding connection is effectively pruned
# Model Architecture

A fully connected neural network:

Input: 32 × 32 × 3 (CIFAR-10 images)
Hidden layers:
512 units
256 units
Output: 10 classes

Custom layer used:

PrunableLinear (instead of nn.Linear)
# Loss Function

The model is trained using a combined loss:

Total Loss = CrossEntropyLoss + λ × SparsityLoss

Where:

CrossEntropyLoss → ensures classification performance
SparsityLoss → L1 norm of all gate values

This encourages the network to minimize unnecessary connections.
