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
Loss Function

Total Loss = CrossEntropyLoss + λ × SparsityLoss

Where:

CrossEntropyLoss → classification objective
SparsityLoss → L1 norm of gate values

This encourages the model to reduce unnecessary connections.

# Results
Lambda=1e-05, Accuracy=50.54%, Sparsity=71.39%
Lambda=0.0001, Accuracy=50.51%, Sparsity=71.56%
Lambda=0.001, Accuracy=50.73%, Sparsity=72.29%

# Observations

<img width="752" height="579" alt="image" src="https://github.com/user-attachments/assets/b3996415-9cf0-4163-9662-37b24ad480e5" />

For higher λ values, the model achieves extreme sparsity (71.56%)
However, accuracy drops to ~50.51%, equivalent to random guessing
This indicates that over-regularization removes all useful connections

# Gate Distribution
Gate values collapse close to 0
This results in complete pruning of the network
The model loses its ability to learn meaningful patterns

# Key Insight

This experiment highlights a critical trade-off:

Low λ → insufficient pruning
High λ → excessive pruning (model collapse)

Effective pruning requires careful tuning of λ to balance performance and sparsity

# Possible Improvements
Use intermediate λ values for balanced pruning
Scale gate scores before sigmoid for sharper gating
Initialize gates with negative bias
Train longer for better convergence
Explore structured pruning or alternative regularization


