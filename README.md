# Sculpting Subspaces: Data-Free Gradient Projection

![Project Hero](assets/subspace_hero.png)
*> Note: 3D Visualization of Gradient Projection onto Null Subspace (Placeholder)*

### "Data-Free Gradient Projection for Continual Learning in CNNs."

## 1. Project Overview
This repository implements an optimized weight-space projection framework for ResNet-18, successfully mitigating catastrophic forgetting on the Split-CIFAR-10 benchmark. By utilizing Randomized SVD (RSVD) and Adaptive Rank Selection, we achieve superior retention and computational efficiency compared to standard spectral methods. The core mechanism involves projecting gradients onto the null space of previous tasks' importance matrices to protect critical features without storing replay data.

## 2. Key Innovations
We extend the standard Gradient Projection Memory (GPM) framework with several novel decomposition strategies:

| Method | Description | Key Advantage |
| :--- | :--- | :--- |
| **Randomized SVD (RSVD)** | Approximates the singular value decomposition using random sampling. | **Speed & Scale**: Drastically reduces subspace computation time for high-dimensional layers compared to deterministic SVD. |
| **Adaptive SVD** |  Dynamically selects rank based on `Input-Output Similarity` sensitivity thresholds (`mrr`, `trr`). | **Precision**: Allocates protection capacity intelligently, preserving more parameters for flexible plasticity in later tasks. |
| **Pivoted QR** | Uses QR decomposition with column pivoting to identify orthogonal bases. | **Stability**: Offers a numerical alternative to SVD, often robust in different weight distributions. |
| **Magnitude Pruning** | Projects gradients by masking the smallest magnitude weights. | **Baseline**: A simple, sparsity-based baseline for comparison. |

## 3. Benchmarks & Results
Our experiments on Split-CIFAR-10 demonstrate that **Randomized SVD (RSVD)** yields the best trade-off between plasticity (learning new tasks) and retention (remembering old tasks), outperforming the Naive baseline significantly.

![Leaderboard Graph](assets/retention_leaderboard.png)
*> Benchmark: Average Retention Accuracy @ 4 Epochs (RSVD denotes highest performance)*

**Key Findings:**
*   **Naive Finetuning** suffers from severe catastrophic forgetting (~20% retention).
*   **Standard SVD** improves retention (~66%) but incurs high computational cost.
*   **RSVD** achieves comparable or superior retention (~68%+) with a fraction of the compute time.
*   **Adaptive SVD** offers competitive performance by automatically tuning the "stiffness" of the model.

## 4. Installation & Usage
To reproduce the experiments:

```bash
# Clone the repository
git clone https://github.com/Dan1k99/deep_learning_project.git
cd deep_learning_project

# Install dependencies
pip install -r requirements.txt

# Run the main experiment suite
# (Note: This runs Naive, SVD, Adaptive, RSVD, and QR experiments sequentially)
python main_experiment.ipynb 
```

## 5. Repository Structure
*   `src/decompositions.py`: Core logic for all subspace projectors (SVD, RSVD, QR, Adaptive).
*   `src/models.py`: ResNet-18 architecture definition.
*   `src/trainer.py`: Training loops for Baseline (Naive) and Constrained (Projected) optimization.
*   `main_experiment.ipynb`: The orchestrator notebook for running benchmarks and visualizing results.
