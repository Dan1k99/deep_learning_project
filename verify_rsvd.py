
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("c:/Users/dani9/.gemini/antigravity/scratch/deep_learning_project"))

from src.decompositions import RSVDProjector

def test_rsvd_stability():
    print("--- Testing RSVD Stability & Accuracy ---")
    torch.manual_seed(42)
    
    # 1. Create a matrix with rapidly decaying singular values
    # Construction: A = U S V^T
    m, n = 100, 100
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    
    # Singular values decaying
    S = torch.logspace(0, -4, steps=min(m, n))
    S_diag = torch.zeros(m, n)
    for i in range(min(m, n)):
        S_diag[i, i] = S[i]
        
    A = U @ S_diag @ V.T
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Condition number: {S[0]/S[-1]:.2e}")
    
    # 2. Initialize RSVD Projector
    # rank_fraction = 0.5 means k=50
    projector = RSVDProjector(rank_fraction=0.5, p=10, q=2)
    
    # Mock a model layer 
    class MockLayer:
        def __init__(self, weight):
            self.weight = weight
            
    class MockModel(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.layer = torch.nn.Parameter(weight)
            # Hack to match named_parameters expectations in compute_subspaces
            # The projector looks for 'weight' in name and dim 2 or 4.
            # We can just create a dummy structure or modify the test to use the method directly if possible.
            # But compute_subspaces iterates model.named_parameters().
            # Let's just patch the method logic or create a real structure.
            self.fc = torch.nn.Linear(n, m, bias=False)
            self.fc.weight.data = A # Linear stores weight as (out, in) = (m, n)
            
    model = MockModel(A)
    
    # 3. specific test for compute_subspaces internal logic
    # We want to see if it runs and produces U, V
    projector.compute_subspaces(model)
    
    if 'fc.weight' in projector.projections:
        U_rsvd, V_rsvd = projector.projections['fc.weight']
        print("RSVD successful. Shapes:", U_rsvd.shape, V_rsvd.shape)
        
        # 4. Compare with specific Target Rank k=50
        k = 50
        
        # Reconstruct approximation
        # U (m, k) @ V.T (k, n) -> No, V_rsvd is (k, n) based on implementation?
        # Implementation: V_keep = Vh[:k, :].T -> (k, n).T -> (n, k).
        # Wait, let's check implementation again.
        # "V_keep = Vh[:k, :].T" 
        # Vh is (l, n). Vh[:k, :] is (k, n). .T is (n, k).
        # So U (m, k), V (n, k).
        # Approx = U @ V.T ?
        # Projector use: inner = U.T @ G @ V.
        # Reconstruct = U @ inner @ V.T.
        # So A_approx = U @ S_k @ V.T ?
        # The projector creates a basis. It doesn't store sigma.
        # But we can check if the subspace covers the top-k singular vectors.
        
        # Let's check subspace angle/orthogonality.
        # Ground Truth SVD
        U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)
        U_true = U_full[:, :k]
        
        # Compute subspace match
        # Projection of U_true onto U_rsvd
        # norm(U_rsvd.T @ U_true) should be close to sqrt(k) if perfectly aligned?
        # Or magnitude of projection.
        # Better: || (I - U_rsvd U_rsvd^T) U_true || should be small.
        
        P_rsvd = U_rsvd @ U_rsvd.T
        diff = torch.norm(U_true - P_rsvd @ U_true)
        print(f"Subspace Error (Left Singular Vectors): {diff:.4e}")
        
        if diff < 1e-1:
            print("✅ RSVD Subspace matches Ground Truth SVD well.")
        else:
            print("❌ RSVD Subspace deviation is high.")
            
    else:
        print("❌ RSVD failed to produce projections.")

if __name__ == "__main__":
    test_rsvd_stability()
