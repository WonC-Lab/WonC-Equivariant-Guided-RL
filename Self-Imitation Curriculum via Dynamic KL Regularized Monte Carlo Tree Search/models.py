import torch
import torch.nn as nn
import torch.nn.functional as F

class D4GroupAction:
    """
    Implements the action of the Dihedral Group D_4 on 2D grid tensors (B, C, H, W).
    """
    @staticmethod
    def apply_action(x: torch.Tensor, action_idx: int) -> torch.Tensor:
        rot_k = action_idx % 4
        flip  = action_idx // 4
        out = x
        if flip == 1:
            out = torch.flip(out, dims=[-1])  # reflect about vertical axis
        if rot_k > 0:
            out = torch.rot90(out, k=rot_k, dims=[-2, -1])
        return out

    @staticmethod
    def apply_inverse_action(x: torch.Tensor, action_idx: int) -> torch.Tensor:
        rot_k = action_idx % 4
        flip  = action_idx // 4
        out = x
        if rot_k > 0:
            out = torch.rot90(out, k=4 - rot_k, dims=[-2, -1])
        if flip == 1:
            out = torch.flip(out, dims=[-1])
        return out

    @staticmethod
    def get_action_permutation(action_idx: int) -> list:
        """
        Computes the permutation σ_g of the 8 action directions induced by
        group element g = D4[action_idx].
        """
        action_vectors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        rot_k = action_idx % 4
        flip  = action_idx // 4

        perm = []
        for dr, dc in action_vectors:
            if flip == 1:
                dc = -dc
            for _ in range(rot_k):
                dr, dc = -dc, dr

            found = False
            for idx, vec in enumerate(action_vectors):
                if vec == (dr, dc):
                    perm.append(idx)
                    found = True
                    break
            if not found:
                raise ValueError(f"Transformed vector ({dr}, {dc}) not in action space.")
        return perm


class StandardCNN(nn.Module):
    """
    Standard (non-equivariant) CNN backbone.
    Used for baseline comparisons.
    """
    def __init__(self, board_size: int = 13, in_channels: int = 3,
                 num_filters: int = 64, num_layers: int = 3):
        super().__init__()
        self.board_size = board_size

        layers = [
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*layers)

        # Policy Head: 8 action logits
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 8),
        )

        # Value Head: outputs scalar ∈ [-10, 10] (to match navigation scale reward)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


class D4EquivariantNet(nn.Module):
    """
    D4-Equivariant Actor-Critic Policy-Value Network via Orbit Batching.
    Guarantees policy equivariance and value invariance.
    """
    def __init__(self, board_size: int = 13, in_channels: int = 3,
                 num_filters: int = 64, num_layers: int = 3):
        super().__init__()
        self.board_size = board_size
        self.base_net = StandardCNN(board_size, in_channels, num_filters, num_layers)
        self.perms = [D4GroupAction.get_action_permutation(i) for i in range(8)]

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        # 1. Orbit batching: stack 8 symmetry views
        x_batched = torch.cat(
            [D4GroupAction.apply_action(x, i) for i in range(8)], dim=0
        )

        # 2. Single forward pass
        logits_batched, values_batched = self.base_net(x_batched)

        # 3. Reshape separate group dimension
        logits_g = logits_batched.view(8, B, 8)   # (8, B, 8)
        values_g  = values_batched.view(8, B, 1)  # (8, B, 1)

        # 4. Realign policy logits: apply permutation σ_g
        realigned = []
        for i in range(8):
            perm_i = self.perms[i]
            realigned.append(logits_g[i][:, perm_i])  # (B, 8)

        # 5. Project back to average (orbit averaging)
        policy_logits = torch.mean(torch.stack(realigned, dim=0), dim=0)  # (B, 8)
        value         = torch.mean(values_g, dim=0)                       # (B, 1)

        return policy_logits, value

    def get_orbit_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance of value predictions across the 8 transformations in the orbit.
        """
        B = x.shape[0]
        # Orbit batching
        x_batched = torch.cat(
            [D4GroupAction.apply_action(x, i) for i in range(8)], dim=0
        )
        with torch.enable_grad():
            _, values_batched = self.base_net(x_batched)
            values_g = values_batched.view(8, B, 1)
            # Compute variance over orbit dimension (dim=0)
            orbit_var = torch.var(values_g, dim=0).squeeze(-1) # (B,)
        return orbit_var
