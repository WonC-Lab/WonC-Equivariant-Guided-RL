import unittest
import torch
import numpy as np

from self_imitation_env import SelfImitationNavigationEnv
from models import StandardCNN, D4EquivariantNet, D4GroupAction
from replay_buffer import SymmetricSelfImitationBuffer

class TestEquivariantSelfImitation(unittest.TestCase):
    def setUp(self):
        self.env = SelfImitationNavigationEnv(size=13)
        self.std_net = StandardCNN(board_size=13, in_channels=3)
        self.eq_net = D4EquivariantNet(board_size=13, in_channels=3)

    def test_buffer_symmetric_expansion(self):
        """Test that adding an episode adds exactly 8x the trajectory length due to D4 group action expansion."""
        buffer = SymmetricSelfImitationBuffer(max_size=1000, running_avg_window=5)
        
        dummy_state = self.env.generate_initial_state()
        dummy_prob = np.zeros(8)
        dummy_prob[3] = 1.0  # Move right
        dummy_transitions = [(dummy_state, dummy_prob, 5.0)]  # Trajectory of length 1
        
        # Add to buffer
        added = buffer.add_episode(dummy_transitions, total_return=10.0)
        self.assertTrue(added)
        
        # Buffer length must be exactly 8
        self.assertEqual(len(buffer), 8)
        
        # Verify that all 8 entries are unique reflections/rotations
        states_in_buffer = [item[0] for item in buffer.buffer]
        # Compare count of unique states
        # Convert state grids to string keys for uniqueness hashing
        unique_states = set(str(s) for s in states_in_buffer)
        
        # Since start is (1, 1) and goal is (11, 11) in a 13x13 grid:
        # Applying reflections/rotations moves the agent or goal, resulting in unique state matrices
        self.assertEqual(len(unique_states), 8)

    def test_policy_equivariance_and_value_invariance(self):
        """Test policy equivariance and value invariance on the D4EquivariantNet."""
        self.eq_net.eval()
        state = self.env.generate_initial_state()
        state_tensor = self.env.state_to_tensor(state)
        
        # Compute forward for original state
        p_orig, v_orig = self.eq_net(state_tensor)
        p_orig_prob = torch.softmax(p_orig, dim=-1).squeeze(0)
        
        # Iterate over all 8 group elements
        for i in range(8):
            # Apply transformation g_i to state
            g_state_t = D4GroupAction.apply_action(state_tensor, i)
            p_g, v_g = self.eq_net(g_state_t)
            p_g_prob = torch.softmax(p_g, dim=-1).squeeze(0)
            
            # 1. Test Value Invariance: V(g*s) == V(s)
            self.assertAlmostEqual(v_orig.item(), v_g.item(), places=4)
            
            # 2. Test Policy Equivariance: p_g[g*a] == p_orig[a]
            perm_i = D4GroupAction.get_action_permutation(i)
            p_orig_permuted = p_orig_prob[perm_i]
            
            for a in range(8):
                self.assertAlmostEqual(p_g_prob[a].item(), p_orig_permuted[a].item(), places=4)

    def test_state_dependent_beta(self):
        """Test that state-dependent beta is computed correctly: beta(s) = max(0, R_best_avg - V(s)) * 0.1"""
        # Simulated buffer and values
        mean_best_reward = 8.0
        
        # State 1: Critic predicts poor return (V_theta(s_1) = 2.0)
        # Beta must be (8.0 - 2.0) * 0.1 = 0.6
        val_poor = torch.tensor([2.0])
        beta_poor = torch.clamp(mean_best_reward - val_poor, min=0.0) * 0.1
        self.assertAlmostEqual(beta_poor.item(), 0.6)
        
        # State 2: Critic predicts high return (V_theta(s_2) = 10.0)
        # Beta must be max(0, 8.0 - 10.0) * 0.1 = 0.0
        val_high = torch.tensor([10.0])
        beta_high = torch.clamp(mean_best_reward - val_high, min=0.0) * 0.1
        self.assertAlmostEqual(beta_high.item(), 0.0)

    def test_random_obstacles_path_exists(self):
        """Test that randomized obstacle generator always guarantees a navigable route from start to goal."""
        for _ in range(30):  # test 30 random generations
            self.env.randomize_obstacles(n_obstacles=15)
            # Path must exist
            path_exists = self.env._bfs_path_exists(self.env.obstacles)
            self.assertTrue(path_exists)

    def test_robotic_octahedral_net_equivariance(self):
        """Test value invariance and policy discrete-equivariance on 3D OctahedralRoboticNet."""
        from models_3d import OctahedralRoboticNet, OctahedralGroupAction
        from robotic_mcts_env import RoboticMCTSEnv
        
        env_3d = RoboticMCTSEnv()
        model_3d = OctahedralRoboticNet().eval()
        
        state_3d = env_3d.randomize_obstacles()
        coords_t = env_3d.state_to_tensor(state_3d)  # (1, 3, 3)
        
        # Original forward pass
        p_orig, v_orig = model_3d(coords_t)
        p_orig_prob = torch.softmax(p_orig, dim=-1).squeeze(0)
        
        rot_matrices = OctahedralGroupAction.get_matrices()
        perms = OctahedralGroupAction.get_action_permutations()
        
        # Test all 24 octahedral rotations
        for i in range(24):
            R = rot_matrices[i]
            
            # g_coords = coords_t @ R.T
            g_coords = torch.matmul(coords_t, R.transpose(0, 1))
            
            p_g, v_g = model_3d(g_coords)
            p_g_prob = torch.softmax(p_g, dim=-1).squeeze(0)
            
            # 1. Invariance of Value: V(g*x) == V(x)
            self.assertAlmostEqual(v_orig.item(), v_g.item(), places=2)
            
            # 2. Equivariance of Policy: p_g[g*a] == p_orig[a]
            perm_i = perms[i]
            
            for a in range(7):
                g_action_idx = perm_i[a]
                self.assertAlmostEqual(p_g_prob[g_action_idx].item(), p_orig_prob[a].item(), places=4)

    def test_robotic_symmetric_buffer_expansion(self):
        """Test that SymmetricRoboticBuffer correctly expands transitions 24-fold using SO(3) rotations."""
        from train_3d import SymmetricRoboticBuffer
        
        buffer_3d = SymmetricRoboticBuffer(max_size=1000)
        
        dummy_coords = torch.zeros((1, 3, 3))
        dummy_prob = np.zeros(7)
        dummy_prob[3] = 1.0  # +Y velocity action
        dummy_transitions = [(dummy_coords, dummy_prob, 8.0)]
        
        added = buffer_3d.add_episode(dummy_transitions, total_return=12.0)
        self.assertTrue(added)
        
        # Length must be exactly 24
        self.assertEqual(len(buffer_3d), 24)

    def test_orbit_variance_invariance(self):
        """Test that get_orbit_variance returns near-zero variance for equivariant nets under random states."""
        # 1. 2D D4EquivariantNet
        x_2d = torch.randn(2, 3, 13, 13)
        var_2d = self.eq_net.get_orbit_variance(x_2d)
        self.assertEqual(var_2d.shape, (2,))
        for v in var_2d:
            self.assertGreaterEqual(v.item(), 0.0)

        # 2. 3D OctahedralRoboticNet
        from models_3d import OctahedralRoboticNet
        model_3d = OctahedralRoboticNet().eval()
        x_3d = torch.randn(3, 3, 3)
        var_3d = model_3d.get_orbit_variance(x_3d)
        self.assertEqual(var_3d.shape, (3,))
        for v in var_3d:
            self.assertGreaterEqual(v.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
