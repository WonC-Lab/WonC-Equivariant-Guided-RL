import numpy as np
import torch
import torch.optim as optim
from autonomous_env import AutonomousNavigationEnv
from equivariant_models import D4EquivariantNet
from mcts_actor_critic import ActorCriticMCTS
from heuristic_guided_loss import HeuristicGuidedLoss

class SymmetricNavEnvAdapter:
    """
    Adapts the 8-directional AutonomousNavigationEnv to a 13x13 (169 actions)
    interface to natively interoperate with the D4EquivariantNet and MCTS.
    """
    def __init__(self, base_env):
        self.base_env = base_env
        self.size = base_env.size

    def clone_state(self, state):
        return self.base_env.clone_state(state)

    def step(self, state, action, turn=1):
        """
        Translates a flat 13x13 grid action index (0-168) into
        the corresponding 8-directional index (0-7), then steps the base environment.
        """
        r, c = self.base_env.get_agent_pos(state)
        ar = action // self.size
        ac = action % self.size
        
        # Calculate offset direction
        dr, dc = ar - r, ac - c
        
        # Find matching action vector index (0-7)
        action_idx = 0
        for idx, vec in enumerate(self.base_env.action_vectors):
            if vec == (dr, dc):
                action_idx = idx
                break
                
        return self.base_env.step(state, action_idx, turn)

    def check_game_over(self, state, turn=1):
        return self.base_env.check_game_over(state, turn)

    def get_valid_actions(self, state, turn=1):
        """
        Converts 8-directional valid indices into 169 flat coordinates.
        """
        r, c = self.base_env.get_agent_pos(state)
        valid_dirs = self.base_env.get_valid_actions(state, turn)
        
        flat_valid_actions = []
        for d_idx in valid_dirs:
            dr, dc = self.base_env.action_vectors[d_idx]
            flat_valid_actions.append((r + dr) * self.size + (c + dc))
            
        return flat_valid_actions

    def state_to_tensor(self, state, turn=1):
        return self.base_env.state_to_tensor(state, turn)

    def get_heuristic_policy_flat(self, state):
        """
        Converts the 8-directional heuristic policy into 169-dimensional policy target.
        """
        r, c = self.base_env.get_agent_pos(state)
        probs_8 = self.base_env.get_heuristic_policy(state)
        
        flat_probs = np.zeros(self.size * self.size, dtype=np.float32)
        for i, (dr, dc) in enumerate(self.base_env.action_vectors):
            nr, nc = r + dr, c + dc
            if self.base_env.in_bounds(nr, nc):
                flat_probs[nr * self.size + nc] = probs_8[i]
                
        # Re-normalize just in case
        total_p = np.sum(flat_probs)
        if total_p > 0:
            flat_probs /= total_p
        return flat_probs


def train_agent():
    print("=============================================================")
    print(" Starting Symmetric & Heuristic-Guided Navigation Training...")
    print("=============================================================")

    # Initialize environment, network, and search algorithms
    base_env = AutonomousNavigationEnv(size=13)
    env = SymmetricNavEnvAdapter(base_env)
    
    # 13x13 grid board, 3 input channels, 32 channels conv layers, 2 layers
    model = D4EquivariantNet(board_size=13, in_channels=3, num_filters=32, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    mcts = ActorCriticMCTS(model=model, c_puct=1.4)
    loss_fn = HeuristicGuidedLoss(beta_start=1.0, beta_decay=0.96, beta_min=0.05)

    num_episodes = 25
    max_steps_per_episode = 25
    gamma = 0.95

    success_count = 0

    for ep in range(1, num_episodes + 1):
        state = base_env.generate_initial_state()
        
        episode_states = []
        episode_actions = []
        episode_heuristics = []
        
        step = 0
        game_over = False
        winner = None

        # Simulation loop
        while not game_over and step < max_steps_per_episode:
            # 1. Run MCTS to obtain action visit counts (simulating high-quality search)
            actions, mcts_probs = mcts.get_action_probabilities(
                state, current_turn=1, game_env=env, num_searches=30, temp=1.0
            )
            
            # Map search distribution back to 169 size
            search_policy_probs = np.zeros(13 * 13, dtype=np.float32)
            for act, prob in zip(actions, mcts_probs):
                search_policy_probs[act] = prob

            # Generate heuristic target for training guidance
            heuristic_probs = env.get_heuristic_policy_flat(state)

            # Record state data
            state_tensor = env.state_to_tensor(state, turn=1)
            episode_states.append(state_tensor)
            
            # Sample action based on MCTS distribution
            chosen_action = np.random.choice(actions, p=mcts_probs)
            episode_actions.append(chosen_action)
            episode_heuristics.append(heuristic_probs)

            # Take step
            state, _ = env.step(state, chosen_action, turn=1)
            game_over, winner = env.check_game_over(state, turn=1)
            step += 1

        # Calculate rewards and returns (Discounted Return G_t)
        if winner == 1:
            final_reward = 1.0
            success_count += 1
            result_str = "SUCCESS (Goal Reached)"
        elif winner == 2:
            final_reward = -1.0
            result_str = "CRASHED (Obstacle/Out of bounds)"
        else:
            final_reward = -0.2
            result_str = "TIMEOUT (Max steps exceeded)"

        # Generate returns vector
        returns = []
        g = final_reward
        for _ in reversed(episode_states):
            returns.append(g)
            g *= gamma
        returns.reverse()

        # Update neural network parameters (Training step)
        if episode_states:
            model.train()
            optimizer.zero_grad()

            # Stack training batch tensors
            batch_states = torch.cat(episode_states, dim=0) # (Batch, 3, 13, 13)
            batch_actions = torch.tensor(episode_actions, dtype=torch.long)
            batch_returns = torch.tensor(returns, dtype=torch.float32)
            batch_heuristics = torch.tensor(np.array(episode_heuristics), dtype=torch.float32)

            # Forward propagation
            logits, values = model(batch_states)

            # Loss computation
            loss, rl_loss_val, kl_loss_val = loss_fn(
                logits, batch_actions, batch_returns, batch_heuristics
            )

            # Backward propagation
            loss.backward()
            optimizer.step()
            loss_fn.decay_beta()

            print(f"Episode {ep:02d}/{num_episodes} | Steps: {step:02d} | Result: {result_str:<32} | Loss: {loss.item():.4f} | Beta: {loss_fn.beta:.4f}")
        else:
            print(f"Episode {ep:02d}/{num_episodes} | Steps: {step:02d} | Result: Empty Episode")

    print("=============================================================")
    print(f" Training Completed. Success Rate: {success_count / num_episodes * 100:.1f}%")
    print("=============================================================")


if __name__ == "__main__":
    train_agent()
