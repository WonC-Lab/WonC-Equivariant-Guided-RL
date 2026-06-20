import torch
import torch.nn as nn
import torch.nn.functional as F

class D4GroupAction:
    """
    Implements the Dihedral Group D_4 actions (8 symmetries: 4 rotations & 4 reflections)
    for 2D grid tensors of shape (Batch, Channel, Height, Width).
    """

    @staticmethod
    def apply_action(x, action_idx):
        """
        Applies one of the 8 group actions of D_4 on a grid tensor.
        action_idx range: 0 to 7.
        """
        # Symmetries:
        # 0: identity, 1: rot90, 2: rot180, 3: rot270
        # 4: flip, 5: flip + rot90, 6: flip + rot180, 7: flip + rot270
        rot_k = action_idx % 4
        flip = action_idx // 4

        out = x
        if flip == 1:
            out = torch.flip(out, dims=[-1])  # horizontal flip
        if rot_k > 0:
            out = torch.rot90(out, k=rot_k, dims=[-2, -1])
        return out

    @staticmethod
    def apply_inverse_action(x, action_idx):
        """
        Applies the mathematical inverse group action to realign features to original orientation.
        """
        rot_k = action_idx % 4
        flip = action_idx // 4

        out = x
        # Inverse rotation is k = 4 - rot_k
        if rot_k > 0:
            out = torch.rot90(out, k=4 - rot_k, dims=[-2, -1])
        if flip == 1:
            out = torch.flip(out, dims=[-1])
        return out


class EquivariantConv2d(nn.Module):
    """
    D_4 Equivariant Convolutional Layer.
    Applies standard Conv2d over all 8 group-transformed views of the input,
    realigns the resulting feature maps using inverse transforms, and averages them.
    Guarantees: f(g * x) = g * f(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        # Share weights across all 8 symmetric views
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        batch_size, in_c, h, w = x.shape
        outputs = []

        # 1. Transform input under 8 symmetries, apply convolution, then inverse-transform back
        for i in range(8):
            transformed_input = D4GroupAction.apply_action(x, i)
            features = self.conv(transformed_input)
            realigned_features = D4GroupAction.apply_inverse_action(features, i)
            outputs.append(realigned_features)

        # 2. Aggregate by averaging (group projection)
        stacked_features = torch.stack(outputs, dim=0) # Shape: (8, Batch, OutChannels, H, W)
        equivariant_features = torch.mean(stacked_features, dim=0)
        return equivariant_features


class D4EquivariantNet(nn.Module):
    """
    Symmetric Actor-Critic Policy-Value Network using D_4 Equivariant convolutions.
    Inputs: (B, Channels, H, W)
    Outputs:
        - policy: (B, H * W) probability distribution (Symmetric Action Space)
        - value: (B, 1) scalar value (Invariant under symmetries)
    """
    def __init__(self, board_size=13, in_channels=3, num_filters=64, num_layers=3):
        super().__init__()
        self.board_size = board_size

        # Stack equivariant convolutional blocks
        layers = [
            EquivariantConv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        ]
        for _ in range(num_layers - 1):
            layers += [
                EquivariantConv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ]
        self.backbone = nn.Sequential(*layers)

        # Policy Head (Equivariant Output via Fully Convolutional Layers)
        self.policy_head = nn.Sequential(
            EquivariantConv2d(num_filters, 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            EquivariantConv2d(2, 1, kernel_size=1, padding=0),
            nn.Flatten()
        )

        # Value Head (Symmetry Invariant Output: f(g*x) = f(x))
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.backbone(x)

        # Policy computation
        policy_logits = self.policy_head(features)

        # Value computation
        value = self.value_head(features)

        return policy_logits, value


# Simple verification
if __name__ == "__main__":
    print("Testing D4 Group Actions and Equivariant Net...")
    model = D4EquivariantNet(board_size=13, in_channels=3, num_filters=32, num_layers=2)
    model.eval()

    # Create a random board state
    x = torch.randn(1, 3, 13, 13)

    # Apply a rot90 transformation
    action_idx = 1
    gx = D4GroupAction.apply_action(x, action_idx)

    # Model inference on original and transformed input
    with torch.no_grad():
        p, v = model(x)
        gp, gv = model(gx)

    # 1. Invariance check for value head: V(g * x) = V(x)
    print(f"Original Value: {v.item():.6f} | Transformed Value: {gv.item():.6f}")
    assert torch.allclose(v, gv, atol=1e-5), "Value Invariance check failed!"
    print("OK: Value head is invariant.")

    # 2. Equivariance check for policy head: P(g * x) = g * P(x)
    # Reshape policy vector to 2D grid to verify rotation
    p_grid = p.view(1, 1, 13, 13)
    gp_grid = gp.view(1, 1, 13, 13)
    transformed_p_grid = D4GroupAction.apply_action(p_grid, action_idx)

    max_diff = torch.max(torch.abs(gp_grid - transformed_p_grid)).item()
    print(f"Policy Equivariance Max Difference: {max_diff:.6f}")
    assert max_diff < 1e-4, "Policy Equivariance check failed!"
    print("OK: Policy head is equivariant under D4.")
