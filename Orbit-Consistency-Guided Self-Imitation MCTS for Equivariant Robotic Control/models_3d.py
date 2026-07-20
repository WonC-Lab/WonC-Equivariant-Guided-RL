import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OctahedralGroupAction:
    """
    Implements the 3D Octahedral Rotation Group O (subset of SO(3), size 24)
    acting on coordinates of shape (B, 3, 3) (ee_pos, target, obstacle).
    """
    # Pre-generate 24 orthogonal 3x3 rotation matrices
    _matrices = None

    @classmethod
    def get_matrices(cls, device=None):
        if cls._matrices is not None:
            # Move to same device if needed
            return cls._matrices.to(device)
            
        matrices = []
        # Octahedral group O consists of all 24 permutations of axes with sign changes
        # such that the determinant of the matrix is +1 (proper rotations).
        from itertools import permutations, product
        
        for p in permutations([0, 1, 2]):
            for signs in product([1, -1], repeat=3):
                # Construct matrix
                R = np.zeros((3, 3))
                for i, axis in enumerate(p):
                    R[i, axis] = signs[i]
                if np.linalg.det(R) == 1.0:  # det = +1 for proper rotation
                    matrices.append(R)
                    
        cls._matrices = torch.tensor(np.array(matrices), dtype=torch.float32)
        if device is not None:
            cls._matrices = cls._matrices.to(device)
        return cls._matrices

    @classmethod
    def get_action_permutations(cls):
        """
        Precomputes how the 24 rotations permute the 7 discrete velocity action directions.
        Actions:
          0: [0, 0, 0] (Stop)
          1: [+0.5, 0, 0] (+X)
          2: [-0.5, 0, 0] (-X)
          3: [0, +0.5, 0] (+Y)
          4: [0, -0.5, 0] (-Y)
          5: [0, 0, +0.5] (+Z)
          6: [0, 0, -0.5] (-Z)
        """
        action_vectors = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0.0, -0.5, 0.0]),
            np.array([0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, -0.5])
        ]
        
        matrices = cls.get_matrices().cpu().numpy()
        permutations_list = []
        
        for R in matrices:
            perm = []
            for v in action_vectors:
                # Apply rotation R to vector v
                R_v = np.dot(R, v)
                
                # Find matching action vector index
                found = False
                for idx, orig_v in enumerate(action_vectors):
                    if np.allclose(R_v, orig_v, atol=1e-5):
                        perm.append(idx)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Rotated action vector {R_v} not in action set.")
            permutations_list.append(perm)
            
        return permutations_list


class StandardRoboticAC(nn.Module):
    """
    Standard (non-equivariant) 3D Robotic Actor-Critic MLP network.
    Inputs: (B, 3, 3) (ee, target, obstacle stacked coordinates)
    """
    def __init__(self, in_features=9, num_actions=7):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.policy_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        # Flatten x: (B, 3, 3) -> (B, 9)
        x_flat = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x_flat))
        h = F.relu(self.fc2(h))
        
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        
        return policy_logits, value


class OctahedralRoboticNet(nn.Module):
    """
    3D Octahedral Equivariant Actor-Critic Policy-Value Network.
    Uses 24-orbit frame averaging to guarantee value invariance and policy equivariance in R^3.
    """
    def __init__(self, board_size=None, in_channels=None):
        # Standard signature compatibility
        super().__init__()
        self.base_net = StandardRoboticAC(in_features=9, num_actions=7)
        # Precompute rotations and permutations
        self.register_buffer("rot_matrices", OctahedralGroupAction.get_matrices())
        self.register_buffer("perms_tensor", torch.tensor(OctahedralGroupAction.get_action_permutations(), dtype=torch.long))

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        device = x.device
        
        # 1. Vectorized orbit batching using torch.einsum
        rot_matrices = self.rot_matrices.to(device)
        x_rotated = torch.einsum('bim, kjm -> kbij', x, rot_matrices) # (24, B, 3, 3)
        x_batched = x_rotated.contiguous().view(24 * B, 3, 3)
        
        # 2. Parallel forward pass
        logits_batched, values_batched = self.base_net(x_batched)
        
        # 3. Separate orbit dimension
        logits_g = logits_batched.view(24, B, 7)
        values_g = values_batched.view(24, B, 1)
        
        # 4. Vectorized output realignment using torch.gather
        perms_expanded = self.perms_tensor.unsqueeze(1).expand(-1, B, -1)  # (24, B, 7)
        realigned_tensor = torch.gather(logits_g, dim=2, index=perms_expanded) # (24, B, 7)
            
        # 5. Orbit projection (average)
        policy_logits = torch.mean(realigned_tensor, dim=0)  # (B, 7)
        value = torch.mean(values_g, dim=0)                  # (B, 1)
        
        return policy_logits, value

    def get_orbit_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance of value predictions across the 24 rotation views in the orbit.
        """
        B = x.shape[0]
        device = x.device
        rot_matrices = self.rot_matrices.to(device)
        x_rotated = torch.einsum('bim, kjm -> kbij', x, rot_matrices) # (24, B, 3, 3)
        x_batched = x_rotated.contiguous().view(24 * B, 3, 3)
        with torch.enable_grad():
            _, values_batched = self.base_net(x_batched)
            values_g = values_batched.view(24, B, 1)
            # Compute variance over orbit dimension (dim=0)
            orbit_var = torch.var(values_g, dim=0).squeeze(-1) # (B,)
        return orbit_var
