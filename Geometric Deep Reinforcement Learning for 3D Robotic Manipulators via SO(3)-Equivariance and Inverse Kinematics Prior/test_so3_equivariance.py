import unittest
import numpy as np
import torch
from robotic_env import RoboticArm3DEnv
from equivariant_models_3d import VNEquivariantPolicyValueNet, StandardMLPPolicyValueNet

def generate_random_so3_matrix():
    """
    Generates a random 3D rotation matrix (SO(3)) using Rodrigues' rotation formula.
    """
    # 1. Random unit axis vector
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    
    # 2. Random angle in [0, 2*pi]
    theta = np.random.uniform(0, 2 * np.pi)
    
    # 3. Skew-symmetric matrix
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ], dtype=np.float32)
    
    # Rodrigues' formula: R = I + sin(theta)*skew + (1 - cos(theta))*skew^2
    I = np.eye(3, dtype=np.float32)
    R = I + np.sin(theta) * skew + (1.0 - np.cos(theta)) * np.dot(skew, skew)
    return torch.tensor(R)

class TestSO3Equivariance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.env = RoboticArm3DEnv()
        self.vn_net = VNEquivariantPolicyValueNet()
        self.mlp_net = StandardMLPPolicyValueNet()
        
        # Put models in eval mode to disable batch norm updates
        self.vn_net.eval()
        self.mlp_net.eval()

    def test_environment_kinematics(self):
        """
        Tests forward kinematics computations and ensures they match expected workspace bounds.
        """
        # Test default reset
        obs = self.env.reset()
        self.assertEqual(obs.shape, (3, 2))
        
        # End-effector position
        ee_pos = obs[:, 0]
        # Target position
        target = obs[:, 1]
        
        # Base height is 1.0, links are 1.0, 1.0. 
        # Verify the height (z-coord) of the end effector is within reach
        self.assertTrue(0.0 <= ee_pos[2] <= 3.0, f"EE z-coord {ee_pos[2]} out of bounds [0, 3]")
        
        # Test step function with zero action
        next_obs, reward, done, info = self.env.step([0.0, 0.0, 0.0])
        self.assertEqual(next_obs.shape, (3, 2))
        self.assertFalse(done)

    def test_vector_neurons_equivariance(self):
        """
        Tests that the Vector Neuron network is SO(3)-equivariant for action
        and SO(3)-invariant for value.
        """
        # Generate random observation inputs (batch_size=5, channels=2, space=3)
        # Net takes (batch, 3, 2)
        x = torch.randn(5, 3, 2)
        
        # Generate a random rotation matrix R
        R = generate_random_so3_matrix() # Shape: (3, 3)
        
        # Rotate inputs: x_rot = R * x (batch, 3, 2)
        # Using batch matrix multiplication
        x_rot = torch.matmul(R, x)
        
        # Pass original through network
        action, value = self.vn_net(x)
        
        # Pass rotated through network
        action_rot, value_rot = self.vn_net(x_rot)
        
        # 1. Verify Policy Equivariance: action_rot == R * action
        expected_action_rot = torch.matmul(R, action.unsqueeze(-1)).squeeze(-1)
        
        # 2. Verify Value Invariance: value_rot == value
        expected_value_rot = value
        
        # Calculate maximum absolute errors
        action_error = torch.max(torch.abs(action_rot - expected_action_rot)).item()
        value_error = torch.max(torch.abs(value_rot - expected_value_rot)).item()
        
        print(f"\n[Vector Neuron Network] Rotation Equivariance Test Results:")
        print(f"  - Action Equivariance Max Absolute Error: {action_error:.6e}")
        print(f"  - Value Invariance Max Absolute Error:    {value_error:.6e}")
        
        # Allow small tolerance for floating point rounding error (e.g., 1e-5)
        self.assertTrue(action_error < 1e-5, f"Vector Neuron action not equivariant (error: {action_error})")
        self.assertTrue(value_error < 1e-5, f"Vector Neuron value not invariant (error: {value_error})")

    def test_standard_mlp_non_equivariance(self):
        """
        Verifies that standard MLP is NOT equivariant/invariant.
        The rotation mapping should fail because standard MLPs do not enforce structural symmetry.
        """
        x = torch.randn(5, 3, 2)
        R = generate_random_so3_matrix()
        x_rot = torch.matmul(R, x)
        
        # Pass through standard MLP
        action, value = self.mlp_net(x)
        action_rot, value_rot = self.mlp_net(x_rot)
        
        expected_action_rot = torch.matmul(R, action.unsqueeze(-1)).squeeze(-1)
        
        action_error = torch.max(torch.abs(action_rot - expected_action_rot)).item()
        value_error = torch.max(torch.abs(value_rot - value)).item()
        
        print(f"\n[Standard MLP Baseline] Rotation Equivariance Test Results:")
        print(f"  - Action Equivariance Max Absolute Error: {action_error:.6e}")
        print(f"  - Value Invariance Max Absolute Error:    {value_error:.6e}")
        
        # In general, a random MLP on random rotated inputs will have large errors (> 0.05)
        # We assert that standard MLP fails to preserve these properties within a precision of 1e-3
        self.assertTrue(action_error > 1e-3, "Standard MLP unexpectedly satisfied equivariance")
        self.assertTrue(value_error > 1e-3, "Standard MLP unexpectedly satisfied invariance")

if __name__ == '__main__':
    unittest.main()
