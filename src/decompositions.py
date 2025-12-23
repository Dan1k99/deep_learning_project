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
        if tensor.dim() == 4:
            # Conv2d: (out_channels, in_channels, k, k) -> (out_channels, in_channels * k * k)
            return tensor.view(tensor.size(0), -1)
        elif tensor.dim() == 2:
            # Linear: (out_features, in_features)
            return tensor
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

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
        self.projections = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Process only .weight tensors of Linear and Conv2d layers
                if 'weight' in name and (param.dim() == 2 or param.dim() == 4):
                    # Flatten layer weights
                    W_flat = self._reshape_layer(param)
                    
                    # Perform SVD
                    # full_matrices=False returns U (M, K), S (K), Vh (K, N) where K = min(M, N)
                    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
                    
                    # Energy Selection
                    S_sq = S ** 2
                    total_energy = torch.sum(S_sq)
                    cum_energy = torch.cumsum(S_sq, dim=0)
                    
                    # Find k such that cumulative energy >= 0.95 * total_energy
                    # searchsorted returns the index where the value would be inserted to maintain order
                    threshold_energy = total_energy * self.rank_threshold
                    k = torch.searchsorted(cum_energy, threshold_energy).item() + 1
                    
                    # Keep top k components
                    # U: (M, K) -> (M, k)
                    # Vh: (K, N) -> (k, N). Since V = Vh.T, V_keep = Vh[:k, :].T
                    U_keep = U[:, :k]
                    V_keep = Vh[:k, :].T
                    
                    self.projections[name] = (U_keep, V_keep)

    def project_gradient(self, layer_name, grad):
        if layer_name not in self.projections:
            return grad
            
        # U_keep: Left Singular Vectors (shape: Rows x Rank)
        # V_keep: Right Singular Vectors (shape: Cols x Rank)
        U_keep, V_keep = self.projections[layer_name]
        
        # 1. Flatten the gradient to a 2D matrix (shape: Rows x Cols)
        original_shape = grad.shape
        grad_flat = self._reshape_layer(grad)
        
        # 2. Compute the "Core Subspace" component of the gradient (Step A)
        # We project the gradient onto the basis defined by U and V.
        # Math: U^T @ Grad @ V
        # dims: (Rank, Rows) @ (Rows, Cols) @ (Cols, Rank) -> Result: (Rank, Rank)
        inner_term = torch.matmul(torch.matmul(U_keep.T, grad_flat), V_keep)
        
        # 3. Project that component back into the full weight space (Step B)
        # This reconstructs the part of the gradient that lies in the "forbidden" subspace.
        # Math: U @ Inner_Term @ V^T
        # dims: (Rows, Rank) @ (Rank, Rank) @ (Rank, Cols) -> Result: (Rows, Cols)
        forbidden_component = torch.matmul(torch.matmul(U_keep, inner_term), V_keep.T)
        
        # 4. Remove the forbidden component (Orthogonal Projection)
        # Math: Grad_Final = Grad_Original - Grad_Forbidden
        grad_proj = grad_flat - forbidden_component
        
        # 5. Reshape back to original tensor dimensions (e.g., 4D for Conv layers)
        return grad_proj.view(original_shape)

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

class MagnitudePruningProjector(SubspaceProjector):
    """
    Baseline: Freezes weights based solely on magnitude. Serves as a control to test if geometric structure (SVD/QR) offers superior retention compared to simple sparsity.
    """
    def compute_subspaces(self, model):
        # TODO: Implement magnitude-based pruning logic
        # TODO: Identify indices of smallest weights (update) vs largest weights (freeze)
        # TODO: Create binary mask instead of projection matrix
        pass

class AdaptiveSVDProjector(SVDProjector):
    """
    Adaptive SVD: Dynamically selects rank based on layer importance (Input-Output Similarity).
    """
    def __init__(self, mrr=0.1, trr=0.9):
        super().__init__()
        self.mrr = mrr # Minimum Retention Ratio (floor)
        self.trr = trr # Target Retention Ratio (ceiling)
        self.projections = {}

    def compute_subspaces(self, model, dataloader, device):
        self.projections = {}
        
        # --- Step A: Calculate Layer Importance ---
        layer_importance = {}
        hooks = []
        activations = {}

        def get_activation_hook(name):
            def hook(model, input, output):
                # Input is a tuple, take first element
                activations[name] = (input[0].detach(), output.detach())
            return hook

        # 1. Register Loops
        for name, module in model.named_modules():
            # We care about layers with weights that we decompose
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(get_activation_hook(name + '.weight')))

        # 2. Run Single Batch
        model.eval()
        try:
            with torch.no_grad():
                # Get one batch
                inputs, _ = next(iter(dataloader))
                inputs = inputs.to(device)
                model(inputs)
        finally:
            # 3. Cleanup Hooks (CRITICAL)
            for h in hooks:
                h.remove()

        # 4. Compute Importance
        for name, (X, Y) in activations.items():
            # Handle Dimension Mismatches
            # Spatial Matching for CNN
            if X.dim() == 4 and Y.dim() == 4:
                # Global Avg Pool: (B, C, H, W) -> (B, C)
                X_pooled = X.mean(dim=(2, 3))
                Y_pooled = Y.mean(dim=(2, 3))
            else:
                 X_pooled, Y_pooled = X, Y
            
            # Channel/Size Check
            if X_pooled.numel() == Y_pooled.numel():
                X_flat = X_pooled.flatten().float()
                Y_flat = Y_pooled.flatten().float()
                
                # Cosine Similarity: (A . B) / (|A| |B|)
                similarity = torch.nn.functional.cosine_similarity(X_flat.unsqueeze(0), Y_flat.unsqueeze(0)).item()
                # Normalize to [0, 1] if needed, but cosine is [-1, 1].
                # We assume importance is absolute alignment or positive alignment.
                # User prompted: "Ensure I_l is normalized between 0 and 1. If similarity is negative, consider using absolute value"
                importance = abs(similarity)
            else:
                # Fallback for mismatched channels
                importance = 1.0
            
            layer_importance[name] = importance
            # print(f"Layer {name}: Importance={importance:.4f}")

        # --- Step B & C: Adaptive Rank & SVD ---
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in layer_importance:
                    # Retrieve Importance
                    imp = layer_importance[name]
                    
                    # Calculate Adaptive Ratio
                    # alpha = mrr + I * (trr - mrr)
                    alpha = self.mrr + imp * (self.trr - self.mrr)
                    
                    # Flatten & SVD (Standard Logic)
                    W_flat = self._reshape_layer(param)
                    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
                    
                    # Adaptive Rank Selection
                    N_sv = len(S)
                    k = int(alpha * N_sv)
                    k = max(1, min(k, N_sv)) # Safety Bounds
                    
                    # Construct Projection
                    U_keep = U[:, :k]
                    V_keep = Vh[:k, :].T
                    self.projections[name] = (U_keep, V_keep)