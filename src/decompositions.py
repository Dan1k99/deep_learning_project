import torch
import torch.nn as nn

class SubspaceProjector:
    """
    Base class for all decomposition techniques.
    Handles the storage of projection matrices.
    """
    def __init__(self, rank_threshold=0.95):
        self.rank_threshold = rank_threshold
        self.projections = {} # Stores {layer_name: projection_matrix}

    def _reshape_layer(self, tensor):
        """
        Macro: Flattens 4D Conv tensors (Out, In, K, K) into 2D matrices[cite: 36].
        """
        pass

    def compute_subspaces(self, model):
        """
        Macro: Iterates over model layers and computes the 'Safe Subspace' 
        using the specific decomposition technique.
        """
        pass

    def project_gradient(self, layer_name, grad):
        """
        Macro: Applies the projection constraint to the gradient[cite: 60].
        formula: grad_clean = grad - projection_matrix @ grad
        """
        pass

class SVDProjector(SubspaceProjector):
    """
    Implementation of the Paper's Baseline (Standard SVD) [cite: 37-40].
    """
    def compute_subspaces(self, model):
        # TODO: Implement torch.linalg.svd logic here
        # TODO: Select top-k singular vectors based on energy
        # TODO: Store U @ U.T matrices
        pass

class QRProjector(SubspaceProjector):
    """
    Implementation of Experiment 2 (QR Decomposition) [cite: 41-45].
    """
    def compute_subspaces(self, model):
        # TODO: Implement torch.linalg.qr logic here
        # TODO: Select first k columns of Q
        pass

class RSVDProjector(SubspaceProjector):
    """
    Implementation of Experiment 3 (Randomized SVD) [cite: 46-50].
    """
    def compute_subspaces(self, model):
        # TODO: Implement randomized SVD (using sklearn or torch approximation)
        pass

class NMFProjector(SubspaceProjector):
    """
    Implementation of Experiment 4 (Non-Negative Matrix Factorization) [cite: 51-55].
    """
    def compute_subspaces(self, model):
        # TODO: Handle non-negativity constraint (abs)
        # TODO: Implement NMF logic
        pass