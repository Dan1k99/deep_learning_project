import torch
import torch.nn as nn

class SubspaceProjector:
    """
    Base class for all decomposition-based subspace projection techniques.
    Maintains a dictionary of per-layer projection matrices computed during
    the subspace identification phase.
    """
    def __init__(self, rank_threshold=0.95):
        self.rank_threshold = rank_threshold
        self.projections = {}  # Maps layer names to their corresponding projection matrices

    def _reshape_layer(self, tensor):
        """
        Reshapes a weight tensor into a 2D matrix suitable for matrix decomposition.
        4D Conv2d tensors of shape (Out, In, K, K) are flattened to (Out, In*K*K);
        2D Linear tensors are returned unchanged.
        """
        if tensor.dim() == 4:
            # Conv2d weights: (out_channels, in_channels, k, k) -> (out_channels, in_channels * k * k)
            return tensor.view(tensor.size(0), -1)
        elif tensor.dim() == 2:
            # Linear weights: (out_features, in_features), already in the required 2D form
            return tensor
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

    def compute_subspaces(self, model):
        """
        Abstract interface: iterates over model layers and populates self.projections
        with the 'safe subspace' representation computed by the concrete subclass.
        """
        pass

    def project_gradient(self, layer_name, grad):
        """
        Abstract interface: applies the orthogonal projection constraint to a gradient tensor.
        The projection removes the component of the gradient that lies within the
        previously identified subspace, following: grad_clean = grad - P @ grad.
        """
        pass

class SVDProjector(SubspaceProjector):
    """
    Implements the paper's baseline subspace projection method using standard
    Truncated Singular Value Decomposition (SVD). The retained subspace is
    determined by an energy threshold that captures a specified fraction of
    the total spectral energy of each weight matrix.
    """
    def compute_subspaces(self, model):
        self.projections = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Restricted to weight tensors of Conv2d and Linear layers
                if 'weight' in name and (param.dim() == 2 or param.dim() == 4):
                    # Flattens the weight tensor to a 2D matrix for decomposition
                    W_flat = self._reshape_layer(param)

                    # Computes the economy SVD; full_matrices=False yields
                    # U (M, K), S (K,), Vh (K, N) where K = min(M, N)
                    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)

                    # Computes the squared singular values (spectral energy per component)
                    S_sq = S ** 2
                    total_energy = torch.sum(S_sq)
                    cum_energy = torch.cumsum(S_sq, dim=0)

                    # Identifies the minimum rank k such that the cumulative energy
                    # meets or exceeds the specified rank_threshold fraction of total energy
                    threshold_energy = total_energy * self.rank_threshold
                    k = torch.searchsorted(cum_energy, threshold_energy).item() + 1

                    # Retains the top-k left and right singular vectors, which together
                    # span the dominant subspace of the weight matrix.
                    # U: (M, K) -> (M, k) | Vh: (K, N) -> V_keep: (N, k) via transposition
                    U_keep = U[:, :k]
                    V_keep = Vh[:k, :].T

                    self.projections[name] = (U_keep, V_keep)

    def project_gradient(self, layer_name, grad):
        if layer_name not in self.projections:
            return grad

        # U_keep: left singular vectors spanning the row subspace  (Rows x Rank)
        # V_keep: right singular vectors spanning the column subspace (Cols x Rank)
        U_keep, V_keep = self.projections[layer_name]

        # Flattens the gradient to a 2D matrix matching the decomposed weight shape
        original_shape = grad.shape
        grad_flat = self._reshape_layer(grad)

        # Projects the gradient onto the dominant subspace (Step A):
        # computes the low-rank coordinate representation of the gradient.
        # Math: U^T @ Grad @ V  |  dims: (Rank, Rows) @ (Rows, Cols) @ (Cols, Rank) -> (Rank, Rank)
        inner_term = torch.matmul(torch.matmul(U_keep.T, grad_flat), V_keep)

        # Reconstructs the forbidden subspace component in the full weight space (Step B):
        # this is the portion of the gradient aligned with the Task A subspace.
        # Math: U @ Inner_Term @ V^T  |  dims: (Rows, Rank) @ (Rank, Rank) @ (Rank, Cols) -> (Rows, Cols)
        forbidden_component = torch.matmul(torch.matmul(U_keep, inner_term), V_keep.T)

        # Removes the forbidden component via orthogonal projection, yielding a gradient
        # that is constrained to the complement of the Task A subspace.
        # Math: Grad_Final = Grad_Original - Grad_Forbidden
        grad_proj = grad_flat - forbidden_component

        # Restores the original tensor shape (e.g., 4D for Conv2d layers)
        return grad_proj.view(original_shape)

class QRProjector(SubspaceProjector):
    """
    Implements subspace projection via Column-Pivoted QR Decomposition (Experiment 4).
    Column pivoting reorders columns by decreasing importance, allowing the leading
    columns of Q to span the most informative directions of the weight matrix.
    """
    def __init__(self, rank_fraction=0.5):
        super().__init__()
        self.rank_fraction = rank_fraction
        self.projections = {}

    def compute_subspaces(self, model, dataloader=None, device='cpu'):
        print("Computing QR subspaces (Pivoted) using Scipy...")
        import scipy.linalg

        for name, param in model.named_parameters():
            if 'weight' in name and (param.dim() == 2 or param.dim() == 4):
                # Flattens the weight tensor to a 2D matrix for decomposition
                W = self._reshape_layer(param)

                # Applies column-pivoted QR via scipy.linalg.qr, which is used in preference
                # to torch.linalg.qr as pivot support is version-dependent in PyTorch.
                # Factorization: W @ P = Q @ R, where P is the column permutation.
                W_np = W.detach().cpu().numpy()
                Q_np, R_np, P_indices = scipy.linalg.qr(W_np, mode='economic', pivoting=True)

                # Determines the truncation rank as a fixed fraction of the matrix's full rank
                full_rank = min(W_np.shape)
                k = int(self.rank_fraction * full_rank)
                k = max(1, k)

                # Retains the leading k columns of Q, which span the most important
                # directions as ordered by the column pivot strategy
                Q_k = Q_np[:, :k]

                # Constructs the orthogonal projection matrix P = Q_k @ Q_k^T,
                # which projects any vector onto the retained k-dimensional subspace
                P_proj_np = Q_k @ Q_k.T

                # Stores the projection matrix as a CPU-resident float tensor
                self.projections[name] = torch.from_numpy(P_proj_np).float().to("cpu")

    def project_gradient(self, layer_name, grad):
        if layer_name not in self.projections:
            return grad

        P_proj = self.projections[layer_name].to(grad.device)

        # Flattens the gradient to a 2D matrix matching the decomposed weight shape
        original_shape = grad.shape
        grad_flat = self._reshape_layer(grad)

        # Projects the gradient onto the retained subspace: grad_subspace = P_proj @ grad_flat
        grad_subspace = torch.matmul(P_proj, grad_flat)

        # Removes the subspace-aligned component via orthogonal complement projection:
        # grad_clean = grad - grad_subspace, constraining updates to the null space of the subspace
        grad_clean = grad_flat - grad_subspace

        # Restores the original tensor shape
        return grad_clean.view(original_shape)

class RSVDProjector(SVDProjector):
    """
    Implements subspace projection via Randomized SVD (Experiment 3),
    following the Halko et al. (2011) algorithm for numerically stable
    low-rank approximation. Power iterations are applied to sharpen the
    approximation quality before the final SVD step.
    Inherits project_gradient from SVDProjector, as both methods store
    projections in the same (U_keep, V_keep) format.
    """
    def __init__(self, rank_fraction=0.5, p=10, q=2):
        super().__init__()
        self.rank_fraction = rank_fraction
        self.p = p  # Oversampling parameter: increases the probability of capturing the true range
        self.q = q  # Number of power iterations: improves approximation accuracy for slowly decaying spectra
        self.projections = {}

    def compute_subspaces(self, model):
        self.projections = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and (param.dim() == 2 or param.dim() == 4):
                    # Flattens the weight tensor to a 2D matrix A of shape (m x n)
                    A = self._reshape_layer(param)
                    m, n = A.shape

                    # Target rank k is set as a fraction of the matrix's smaller dimension
                    k = int(min(m, n) * self.rank_fraction)
                    k = max(1, k)

                    # Oversampled sketch size l = k + p; clamped to the matrix's rank
                    # to avoid requesting more columns than the matrix can provide
                    l = k + self.p
                    if l > min(m, n):
                        l = min(m, n)

                    # --- Stage A: Randomized Range Finder ---

                    # Draws a Gaussian random test matrix Omega of shape (n x l)
                    Omega = torch.randn(n, l, device=A.device, dtype=A.dtype)

                    # Forms the initial sample matrix Y = A @ Omega, approximating the range of A
                    Y = torch.matmul(A, Omega)

                    # Applies q power iterations with alternating QR orthogonalization to
                    # sharpen the range approximation; each iteration amplifies the dominant
                    # singular directions relative to the smaller ones.
                    for _ in range(self.q):
                        # Orthogonalizes Y to maintain numerical stability
                        Q_Y, _ = torch.linalg.qr(Y)

                        # Projects onto the row space of A via A^T
                        Z = torch.matmul(A.T, Q_Y)

                        # Orthogonalizes Z before the next forward projection
                        Q_Z, _ = torch.linalg.qr(Z)

                        # Projects back onto the column space of A
                        Y = torch.matmul(A, Q_Z)

                    # Computes the final orthonormal basis Q approximating the range of A
                    Q, _ = torch.linalg.qr(Y)

                    # --- Stage B: Projected SVD ---

                    # Projects A into the low-dimensional space: B = Q^T @ A
                    # dims: (l x m) @ (m x n) -> (l x n); B is small since l << m typically
                    B = torch.matmul(Q.T, A)

                    # Computes the exact SVD of the small projected matrix B
                    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

                    # Recovers the approximate left singular vectors of A: U_approx = Q @ U_hat
                    # dims: (m x l) @ (l x l) -> (m x l)
                    U_approx = torch.matmul(Q, U_hat)

                    # Truncates to the target rank k, retaining the dominant singular directions
                    U_keep = U_approx[:, :k]
                    # Vh is (l x n); V_keep is obtained as the transpose of the first k rows
                    V_keep = Vh[:k, :].T

                    self.projections[name] = (U_keep, V_keep)

class MagnitudePruningProjector(SubspaceProjector):
    """
    Implements gradient masking via Global Unstructured Magnitude Pruning (Experiment 6).
    Weights with the smallest L1 magnitudes are zeroed out globally across all layers
    to achieve a target sparsity level. The resulting binary mask is then applied to
    gradients during Task B training, preventing updates to pruned weight positions.
    """
    def __init__(self, sparsity_target=0.5):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.projections = {}  # Maps parameter names to binary sparsity masks

    def compute_subspaces(self, model):
        import torch.nn.utils.prune as prune

        # Collects all Conv2d and Linear weight tensors eligible for pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        print(f"Pruning {len(parameters_to_prune)} layers with Global Sparsity {self.sparsity_target*100:.1f}%...")

        # Applies global unstructured L1 pruning across all collected parameters simultaneously;
        # the pruning threshold is determined globally so that the overall sparsity target is met.
        # This operation modifies weights in-place, creating weight_mask and weight_orig buffers.
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity_target,
        )

        # Extracts and stores the binary masks, then makes pruning permanent by removing
        # the auxiliary buffers and consolidating the pruned weights into the weight parameter.
        self.projections = {}
        total_zeros = 0
        total_elements = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask.detach().clone()
                    # Finalizes pruning: removes the weight_mask and weight_orig buffers,
                    # leaving only the pruned weight tensor in place.
                    prune.remove(module, 'weight')
                else:
                    # Fallback for layers where the mask buffer is absent (e.g., on re-execution):
                    # reconstructs the mask from the zero pattern of the current weight tensor.
                    mask = (module.weight != 0).float()

                # Stores the mask under the fully qualified parameter name (e.g., 'layer1.0.conv1.weight')
                # to match the keys produced by model.named_parameters() in the training loop.
                if name == "":
                    param_name = "weight"
                else:
                    param_name = name + ".weight"

                self.projections[param_name] = mask

                # Accumulates sparsity statistics for the final diagnostic report
                zeros = (module.weight == 0).sum().item()
                elems = module.weight.numel()
                total_zeros += zeros
                total_elements += elems

        global_sparsity = total_zeros / total_elements
        print(f"Global Sparsity Achieved: {global_sparsity*100:.2f}%")

    def project_gradient(self, layer_name, grad):
        # Enforces gradient sparsity by zeroing out gradient entries at pruned weight positions,
        # ensuring that the optimizer cannot restore weights that were removed during pruning.
        if layer_name in self.projections:
            mask = self.projections[layer_name].to(grad.device)
            return grad * mask
        return grad

class AdaptiveSVDProjector(SVDProjector):
    """
    Extends SVDProjector with a data-driven, layer-wise rank selection strategy.
    The retained rank for each layer is determined by its input-output activation
    similarity: layers that behave more like identity mappings (high cosine similarity
    between input and output) are assigned a higher retention ratio, preserving more
    of their subspace. Importance scores are normalized relative to the layer-wise mean
    to prevent extreme rank collapse or inflation.
    """
    def __init__(self, mrr=0.4, trr=0.95):
        super().__init__()
        self.mrr = mrr  # Minimum Retention Ratio: lower bound on the fraction of singular values retained
        self.trr = trr  # Target Retention Ratio: upper bound, approached for the most important layers
        self.projections = {}

    def compute_subspaces(self, model, dataloader, device):
        self.projections = {}

        # --- Phase A: Activation-Based Importance Scoring ---
        raw_importance = {}
        hooks = []
        activations = {}

        def get_activation_hook(name):
            def hook(model, input, output):
                # Captures the input and output activations of each layer for a single forward pass
                activations[name] = (input[0].detach(), output.detach())
            return hook

        # Registers forward hooks on all Conv2d and Linear layers to capture their activations
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(get_activation_hook(name + '.weight')))

        # Executes a single forward pass on one batch to populate the activation dictionary;
        # hooks are removed in the finally block to avoid side effects on subsequent passes.
        model.eval()
        try:
            with torch.no_grad():
                inputs, _ = next(iter(dataloader))
                inputs = inputs.to(device)
                model(inputs)
        finally:
            for h in hooks:
                h.remove()

        # Computes the cosine similarity between input and output activations for each layer.
        # For 4D CNN tensors, global average pooling reduces spatial dimensions before comparison.
        valid_scores = []
        for name, (X, Y) in activations.items():
            if X.dim() == 4 and Y.dim() == 4:
                # Global average pooling: (B, C, H, W) -> (B, C), collapsing spatial dimensions
                X_pooled = X.mean(dim=(2, 3))
                Y_pooled = Y.mean(dim=(2, 3))
            else:
                X_pooled, Y_pooled = X, Y

            if X_pooled.numel() == Y_pooled.numel():
                X_flat = X_pooled.flatten().float()
                Y_flat = Y_pooled.flatten().float()

                # Cosine similarity in [-1, 1]; its absolute value serves as the importance score,
                # with higher values indicating that the layer preserves its input (identity-like behavior).
                similarity = torch.nn.functional.cosine_similarity(X_flat.unsqueeze(0), Y_flat.unsqueeze(0)).item()
                imp = abs(similarity)

                raw_importance[name] = imp
                valid_scores.append(imp)
            else:
                # Dimension mismatch (e.g., residual expansion layers): importance deferred to mean imputation
                raw_importance[name] = None

        # --- Phase B: Score Normalization and Rank Computation ---

        # Computes the mean importance across all layers with valid scores;
        # used as the normalization baseline and as a fallback for mismatched layers.
        if len(valid_scores) > 0:
            avg_importance = sum(valid_scores) / len(valid_scores)
        else:
            avg_importance = 1.0  # Conservative fallback when no valid scores are available

        print(f"Adaptive SVD: Avg Importance = {avg_importance:.4f} (over {len(valid_scores)} matching layers)")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in raw_importance:
                    raw = raw_importance[name]
                    if raw is None:
                        # Assigns mean-normalized importance (norm_imp = 1.0) to dimension-mismatched layers,
                        # placing them at the center of the retention ratio range as a neutral stance.
                        norm_imp = 1.0
                    else:
                        # Normalizes each layer's importance relative to the population mean,
                        # so that average layers receive a retention ratio near the midpoint of [mrr, trr].
                        norm_imp = raw / avg_importance

                    # Computes the adaptive retention ratio alpha via linear interpolation
                    # between mrr and trr, scaled by the normalized importance score.
                    # alpha is clamped to [mrr, 1.0] to enforce the minimum retention floor
                    # and prevent exceeding full rank.
                    alpha = self.mrr + norm_imp * (self.trr - self.mrr)
                    alpha = max(self.mrr, min(alpha, 1.0))

                    # Flattens the weight tensor and computes its full SVD
                    W_flat = self._reshape_layer(param)
                    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)

                    # Determines the number of singular vectors to retain based on the adaptive ratio,
                    # bounded to [1, N_sv] to guarantee at least one component is always kept.
                    N_sv = len(S)
                    k = int(alpha * N_sv)
                    k = max(1, min(k, N_sv))

                    # Stores the truncated left and right singular vectors as the layer's projection basis
                    U_keep = U[:, :k]
                    V_keep = Vh[:k, :].T
                    self.projections[name] = (U_keep, V_keep)