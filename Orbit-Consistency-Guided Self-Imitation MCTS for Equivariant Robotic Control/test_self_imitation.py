import unittest
import torch
import numpy as np

from self_imitation_env import SelfImitationNavigationEnv
from models import StandardCNN
from mcts import ActorCriticMCTS
from replay_buffer import SymmetricSelfImitationBuffer

class TestSelfImitationFramework(unittest.TestCase):
    def setUp(self):
        self.env = SelfImitationNavigationEnv(size=13)
        self.model = StandardCNN(board_size=13, in_channels=3)
        
    def test_environment_initialization_and_step(self):
        """Test that the environment initializes and processes steps correctly."""
        state = self.env.generate_initial_state()
        self.assertEqual(len(state), 13)
        self.assertEqual(len(state[0]), 13)
        
        # Initial agent position should be (1, 1)
        r, c = self.env.get_agent_pos(state)
        self.assertEqual((r, c), (1, 1))
        
        # Test step - move right (action 3 corresponds to (0, 1))
        next_state, _ = self.env.step(state, 3)
        nr, nc = self.env.get_agent_pos(next_state)
        self.assertEqual((nr, nc), (1, 2))
        
        # Agent's old position should be cleared to 0 (empty)
        self.assertEqual(next_state[1][1], 0)
        # New position should be 1 (agent)
        self.assertEqual(next_state[1][2], 1)
        
    def test_model_forward_pass(self):
        """Test that the neural network receives states and outputs correct shapes."""
        state = self.env.generate_initial_state()
        state_tensor = self.env.state_to_tensor(state)
        
        # Expected input shape (1, 3, 13, 13)
        self.assertEqual(state_tensor.shape, (1, 3, 13, 13))
        
        policy_logits, value = self.model(state_tensor)
        
        # Expected outputs: policy shape (1, 8), value shape (1, 1)
        self.assertEqual(policy_logits.shape, (1, 8))
        self.assertEqual(value.shape, (1, 1))
        
    def test_replay_buffer_filtering(self):
        """Test that the replay buffer filters out sub-average returns."""
        buffer = SymmetricSelfImitationBuffer(max_size=80, running_avg_window=5)
        
        # Add a few baseline episodes
        # Added mock transitions: (state, mcts_prob, return_g)
        dummy_state = self.env.generate_initial_state()
        dummy_prob = np.ones(8) / 8.0
        dummy_transitions = [(dummy_state, dummy_prob, 0.0)]
        
        # First 10 episodes are always added to populate running average baseline
        for i in range(1, 11):
            added = buffer.add_episode(dummy_transitions, total_return=-float(i))
            self.assertTrue(added)
            
        # Running average should be around np.mean([-6, -7, -8, -9, -10]) = -8.0
        running_avg = buffer.get_running_average()
        self.assertAlmostEqual(running_avg, -8.0)
        
        # Add episode with return -9.0 (below average of -8.0). Should NOT be added
        added = buffer.add_episode(dummy_transitions, total_return=-9.0)
        self.assertFalse(added)
        
        # Add episode with return -4.0 (above average of -8.0). Should be added
        added = buffer.add_episode(dummy_transitions, total_return=-4.0)
        self.assertTrue(added)
        
    def test_mcts_distribution(self):
        """Test that MCTS performs simulation search and returns valid probability distributions."""
        state = self.env.generate_initial_state()
        mcts = ActorCriticMCTS(self.model, c_puct=1.5)
        
        actions, probs = mcts.get_action_probabilities(
            state, current_turn=1, game_env=self.env, num_searches=10, temp=1.0
        )
        
        # Valid actions list should not be empty
        self.assertTrue(len(actions) > 0)
        self.assertEqual(len(actions), len(probs))
        
        # Sum of probabilities should be close to 1.0
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        
        # All probabilities should be non-negative
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

if __name__ == '__main__':
    unittest.main()
