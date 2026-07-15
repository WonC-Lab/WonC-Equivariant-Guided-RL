import torch
import torch.nn as nn
import torch.nn.functional as F

class PermutationEquivariantGNN(nn.Module):
    """
    Permutation-Equivariant Policy-Value Network for Multi-Agent RL.
    - Processes batch of observations of shape (B, M, 3, H, W)
    - Employs a Shared CNN for spatial feature extraction per agent.
    - Uses Self-Attention (Transformer Encoder) to share coordination features.
      Since there is no positional encoding, the Transformer operates equivariantly 
      with respect to the permutation of agents.
    - Outputs:
      - Policy Logits: (B, M, 8)
      - Values: (B, M, 1)
    """
    def __init__(self, grid_size=13, in_channels=3, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        
        # Calculate CNN target pooling dimensions
        dummy_h = grid_size // 2 if grid_size >= 8 else grid_size
        dummy_w = grid_size // 2 if grid_size >= 8 else grid_size
        cnn_out_dim = 32 * dummy_h * dummy_w

        # 1. Shared Spatial CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if grid_size >= 8 else nn.Identity(), # Downsample if map is large
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((dummy_h, dummy_w)),
            nn.Flatten()
        )
        
        # Feature projection to d_model
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, d_model),
            nn.ReLU()
        )
        
        # 2. Permutation-Equivariant Multi-Agent Attention Layer
        # batch_first=True expects input shape (B, M, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2 * d_model, 
            dropout=0.0, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Policy and Value Heads
        self.policy_head = nn.Linear(d_model, 8)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, obs):
        """
        obs: (B, M, 3, H, W)
        Returns:
          logits: (B, M, 8)
          values: (B, M, 1)
        """
        B, M, C, H, W = obs.shape
        
        # Flatten B and M for CNN processing
        obs_flat = obs.view(B * M, C, H, W)
        
        # Extract features
        cnn_feats = self.cnn(obs_flat)
        local_embeddings = self.proj(cnn_feats) # (B*M, d_model)
        
        # Reshape to (B, M, d_model) for attention
        local_embeddings = local_embeddings.view(B, M, self.d_model)
        
        # Multi-Agent Coordination via Attention (No positional encoding -> permutation equivariant!)
        coordinated_embeddings = self.transformer(local_embeddings) # (B, M, d_model)
        
        # Reshape for heads
        coordinated_flat = coordinated_embeddings.view(B * M, self.d_model)
        
        # Output heads
        logits_flat = self.policy_head(coordinated_flat) # (B*M, 8)
        values_flat = torch.tanh(self.value_head(coordinated_flat)) # (B*M, 1)
        
        # Reshape back to joint outputs
        logits = logits_flat.view(B, M, 8)
        values = values_flat.view(B, M, 1)
        
        return logits, values

# Verification of Permutation Equivariance
if __name__ == "__main__":
    print("Testing Permutation Equivariance...")
    grid_size = 13
    num_agents = 4
    model = PermutationEquivariantGNN(grid_size=grid_size, num_agents=num_agents)
    model.eval()
    
    # 1. Create a dummy observation: batch=1, agents=4
    obs = torch.randn(1, num_agents, 3, grid_size, grid_size)
    
    # 2. Forward pass with original ordering
    with torch.no_grad():
        orig_logits, orig_vals = model(obs)
    
    # 3. Define a permutation (e.g. swap agent 0 and 2, and agent 1 and 3)
    p_indices = [2, 3, 0, 1]
    permuted_obs = obs[:, p_indices, :, :, :]
    
    # 4. Forward pass with permuted ordering
    with torch.no_grad():
        perm_logits, perm_vals = model(permuted_obs)
        
    # 5. Permute the original outputs for comparison
    expected_perm_logits = orig_logits[:, p_indices, :]
    expected_perm_vals = orig_vals[:, p_indices, :]
    
    # 6. Verify mathematically
    diff_logits = torch.abs(perm_logits - expected_perm_logits).max().item()
    diff_vals = torch.abs(perm_vals - expected_perm_vals).max().item()
    
    print(f"Max difference in policy logits: {diff_logits:.8f}")
    print(f"Max difference in state values: {diff_vals:.8f}")
    
    assert diff_logits < 1e-5, "Permutation Equivariance test failed on policy logits!"
    assert diff_vals < 1e-5, "Permutation Equivariance test failed on state values!"
    print("OK: Permutation Equivariance Verified Successfully!")
