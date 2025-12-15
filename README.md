# Optimizing Subspace Constraints: Comparative Matrix Decomposition Techniques for Continual Learning in ResNet-18

## 🎯 Project Goal
This is a research project comparing different mathematical techniques (Standard SVD, QR Decomposition, Randomized SVD, and Magnitude Pruning) to solve "Catastrophic Forgetting" in Neural Networks. It replicates and upgrades the methodology of the "Sculpting Subspaces" paper by replacing the core SVD engine with other decomposition methods.

## 🏗️ Architecture & File Structure

### 📓 The Orchestrator
- **`main_experiment.ipynb`**: The entry point.
    - Configures experiments.
    - Trains the expert model (Phase 2).
    - Executes the comparative loop (Phase 3 & 4).
    - Produces final results and plots.

### 🐍 Source Directory (`src/`)
- **`src/__init__.py`**: Defines the directory as a Python package.
- **`src/data_utils.py` (Phase 1)**: Handles "Split-CIFAR-10" logic.
    - Divides the dataset into **Task A** (Classes 0-4, old knowledge) and **Task B** (Classes 5-9, new knowledge).
- **`src/models.py` (Phase 2)**: Wraps `torchvision.models.resnet18`.
    - **Crucial**: Modifies the output layer to support 10 classes even when training on the 5-class Task A, facilitating Continual Learning.
- **`src/decompositions.py` (Phase 3 - The ❤️ of the Project)**:
    - Contains the `SubspaceProjector` abstract class.
    - Implements **SVD**, **QR**, **Magnitude Pruning**, and **Randomized SVD**.
    - Calculates the "Safe Subspaces" where weights are allowed to change.
- **`src/trainer.py` (Phase 2 & 4)**: Contains custom training loops.
    - `train_baseline`: Standard training for the Expert model.
    - `train_constrained`: Experimental loop applying **Gradient Projection/Cleaning** intervention before the optimizer step.

### 📁 Checkpoints
- **`checkpoints/`**: Stores trained `task_a_expert.pth` weights to avoid retraining the baseline.

## 🔄 Project Workflow
1.  **Data Setup**: Split CIFAR-10 into Task A and Task B.
2.  **Baseline Training**: Train ResNet-18 on Task A to create the "Expert".
3.  **Subspace Extraction**: Use decomposition techniques (SVD, QR, etc.) to find important weights in the Expert.
4.  **Constrained Training**: Train on Task B while projecting gradients to preserve Task A knowledge.
5.  **Analysis**: Compare performance across different decomposition methods.
