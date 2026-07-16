import torch
import torch.nn as nn
import torch.nn.functional as F

class VNLinear(nn.Module):
    """
    Vector Neuron Linear Layer.
    Maps in_channels vectors in R^3 to out_channels vectors in R^3.
    """
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        # Linear map across channels. Weight matrix size is (out_channels, in_channels)
        # We must disable bias to maintain SO(3) equivariance.
        self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        # x shape: (batch, in_channels, 3)
        # Permute to (batch, 3, in_channels) to apply linear mapping over the channel dimension
        x_t = x.transpose(1, 2)
        out_t = self.map_to_dir(x_t)
        # Permute back to (batch, out_channels, 3)
        return out_t.transpose(1, 2)


class VNLeakyReLU(nn.Module):
    """
    Vector Neuron LeakyReLU.
    For each channel v_c in R^3, projects v_c onto a learned direction q_c.
    If dot(v_c, q_c) >= 0, returns v_c.
    Otherwise, returns the perpendicular component + scaled parallel component.
    """
    def __init__(self, in_channels, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        # Learn a direction for each channel using a linear mapping (no bias)
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        # x shape: (batch, in_channels, 3)
        x_t = x.transpose(1, 2)
        q_t = self.map_to_dir(x_t)
        q = q_t.transpose(1, 2) # (batch, in_channels, 3)

        eps = 1e-6
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q_dir = q / (q_norm + eps)

        # Dot product along the spatial dimension
        proj = torch.sum(x * q_dir, dim=-1, keepdim=True) # (batch, in_channels, 1)

        # If proj < 0, we clamp the projection term to scale it down
        proj_neg = torch.clamp(proj, max=0.0)
        
        # Out = x - (1 - alpha) * proj_neg * q_dir
        # Since proj_neg is negative/zero, this scales down the negative projection by alpha (negative_slope)
        out = x - (1.0 - self.negative_slope) * proj_neg * q_dir
        return out


class VNBatchNorm(nn.Module):
    """
    Vector Neuron Batch Normalization.
    Normalizes vector features using the mean square of their norms.
    """
    def __init__(self, num_features, eps=1e-5):
        super(VNBatchNorm, self).__init__()
        self.eps = eps
        self.register_buffer('running_mean_sq', torch.ones(num_features))

    def forward(self, x):
        # x shape: (batch, num_features, 3)
        if self.training:
            # Norm squared along the 3D space dimension (shape: batch, num_features)
            norm_sq = torch.sum(x**2, dim=-1)
            # Average over the batch dimension
            mean_sq = torch.mean(norm_sq, dim=0) # (num_features,)
            
            with torch.no_grad():
                self.running_mean_sq.copy_(0.9 * self.running_mean_sq + 0.1 * mean_sq)
        else:
            mean_sq = self.running_mean_sq

        # Reshape for broadcasting
        scale = 1.0 / torch.sqrt(mean_sq + self.eps)
        scale = scale.view(1, -1, 1) # (1, num_features, 1)
        return x * scale


class VNEquivariantPolicyValueNet(nn.Module):
    """
    SO(3)-equivariant policy-value network using Vector Neurons.
    Input observation shape: (batch, 3, 2)
      - Row 0: x-coordinates of [ee_pos, target_pos]
      - Row 1: y-coordinates of [ee_pos, target_pos]
      - Row 2: z-coordinates of [ee_pos, target_pos]
    """
    def __init__(self):
        super(VNEquivariantPolicyValueNet, self).__init__()
        
        # Feature extraction layers (3 input channels: ee_pos, target_pos, obstacle_pos)
        self.vn_conv1 = VNLinear(3, 64)
        self.vn_act1 = VNLeakyReLU(64)
        
        self.vn_conv2 = VNLinear(64, 64)
        self.vn_act2 = VNLeakyReLU(64)
        
        # Policy head (equivariant: outputs 1 coordinate vector in R^3)
        self.vn_policy = VNLinear(64, 1)
        
        # Value head (invariant: maps vector norms to scalar)
        self.value_fc1 = nn.Linear(64, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Input x shape: (batch, 3, 3)
        # Permute to (batch, 3, 3) to treat [ee, target, obstacle] as 3 channels of 3D vectors
        x_perm = x.transpose(1, 2)
        
        # Layer 1
        h = self.vn_conv1(x_perm)
        h = self.vn_act1(h)
        
        # Layer 2
        h = self.vn_conv2(h)
        h = self.vn_act2(h) # (batch, 64, 3)
        
        # Equivariant policy action (batch, 1, 3)
        action_vector = self.vn_policy(h)
        # Squeeze channels to get (batch, 3)
        action = action_vector.squeeze(1)
        
        # Invariant value prediction (detach to prevent value gradients from corrupting policy)
        # Compute L2 norm of the vector features across all 64 channels
        # norm shape: (batch, 64)
        feature_norms = torch.norm(h.detach(), dim=-1)
        
        val = F.relu(self.value_fc1(feature_norms))
        value = self.value_fc2(val) # (batch, 1)
        
        return action, value


class StandardMLPPolicyValueNet(nn.Module):
    """
    Standard MLP baseline network (non-equivariant).
    Flattens the observation into R^6 and processes it using standard MLPs.
    """
    def __init__(self):
        super(StandardMLPPolicyValueNet, self).__init__()
        
        # Input dimension is 9 (ee_pos, target_pos, obstacle_pos flattened: 3*3)
        self.shared_fc = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # Input x shape: (batch, 3, 2)
        # Flatten to (batch, 6)
        x_flat = x.reshape(x.size(0), -1)
        
        h = self.shared_fc(x_flat)
        
        action = self.policy_head(h)
        # Detach to prevent value gradients from corrupting policy features
        value = self.value_head(h.detach())
        
        return action, value
