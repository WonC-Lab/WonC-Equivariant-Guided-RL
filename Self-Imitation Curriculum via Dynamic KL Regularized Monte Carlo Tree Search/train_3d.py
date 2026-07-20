import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from robotic_mcts_env import RoboticMCTSEnv
from models_3d import StandardRoboticAC, OctahedralRoboticNet, OctahedralGroupAction
from mcts import ActorCriticMCTS

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SymmetricRoboticBuffer:
    """
    Symmetric Replay Buffer for 3D Robotic Arm.
    Expands each transition 24x using the Octahedral rotation group O in R^3.
    """
    def __init__(self, max_size=10000, running_avg_window=30):
        self.max_size = max_size
        self.buffer = []  # List of tuples: (coords_tensor, mcts_prob, return_g)
        self.episode_returns = []
        self.running_avg_window = running_avg_window
        self.best_episode_returns = []
        self.perms = OctahedralGroupAction.get_action_permutations()

    def get_running_average(self):
        if not self.episode_returns:
            return -10.0
        window = min(len(self.episode_returns), self.running_avg_window)
        return np.mean(self.episode_returns[-window:])

    def get_mean_best_reward(self):
        if not self.best_episode_returns:
            return -2.0
        return np.mean(self.best_episode_returns)

    def add_episode(self, episode_transitions, total_return):
        running_avg = self.get_running_average()
        self.episode_returns.append(total_return)
        
        # Buffer criteria: better than running average or succeeded (total_return > 5.0)
        if len(self.episode_returns) <= 10 or total_return > running_avg or total_return > 5.0:
            self.best_episode_returns.append(total_return)
            
            # Load 24 rotations
            rot_matrices = OctahedralGroupAction.get_matrices()
            
            for coords_tensor, mcts_prob, return_g in episode_transitions:
                # coords_tensor is shape (1, 3, 3)
                for i in range(24):
                    R = rot_matrices[i]
                    
                    # Coordinate transform: coords @ R.T
                    g_coords = torch.matmul(coords_tensor, R.transpose(0, 1))
                    
                    # Action probability distribution permutation
                    perm_i = self.perms[i]
                    g_prob = np.zeros(7)
                    for act_idx, p in enumerate(mcts_prob):
                        g_prob[perm_i[act_idx]] = p
                        
                    self.buffer.append((g_coords, g_prob, return_g))
                    
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]
                
            return True
        return False

    def sample(self, batch_size):
        import random
        if len(self.buffer) < batch_size:
            samples = random.choices(self.buffer, k=batch_size)
        else:
            samples = random.sample(self.buffer, batch_size)
        coords, mcts_probs, returns = zip(*samples)
        return coords, mcts_probs, returns

    def __len__(self):
        return len(self.buffer)

def train_robotic_agent(model_class, num_episodes=600, max_steps=100, mcts_searches=15, alpha=0.1, seed=42, ablation_type="mult", eta=0.05, lambda_val=0.05):
    set_seed(seed)
    env = RoboticMCTSEnv(max_steps=max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    mcts = ActorCriticMCTS(model, c_puct=1.5)
    buffer = SymmetricRoboticBuffer(max_size=15000, running_avg_window=20)
    
    print(f"Starting 3D Robotic training for {model_class.__name__} (seed={seed}, episodes={num_episodes})...")
    for episode in range(1, num_episodes + 1):
        state = env.randomize_obstacles()
        
        episode_transitions = []
        total_undiscounted_reward = 0.0
        
        temp = 1.0 if episode < 50 else 0.5
        
        step = 0
        game_over = False
        winner = None
        
        states_in_episode = []
        actions_taken = []
        mcts_policies = []
        rewards = []
        
        while not game_over and step < max_steps:
            actions, probs = mcts.get_action_probabilities(
                state, current_turn=1, game_env=env, num_searches=mcts_searches, temp=temp
            )
            
            if not actions:
                break
                
            action = np.random.choice(actions, p=probs)
            
            full_prob = np.zeros(7)
            for act, prob in zip(actions, probs):
                full_prob[act] = prob
                
            # Pre-compute coordinates tensor for buffer
            coords_tensor = env.state_to_tensor(state)
            
            states_in_episode.append(coords_tensor)
            actions_taken.append(action)
            mcts_policies.append(full_prob)
            
            next_state, _ = env.step(state, action)
            
            game_over, winner = env.check_game_over(next_state)
            if game_over:
                if winner == 1:
                    step_reward = 10.0
                else:
                    step_reward = -5.0
            else:
                # Soft proximity reward: encourage getting closer to target
                ee_pos = env.forward_kinematics(next_state["theta"])
                target_pos = np.array(next_state["target"])
                dist = np.linalg.norm(ee_pos - target_pos)
                step_reward = -dist - 0.1
                
            rewards.append(step_reward)
            total_undiscounted_reward += step_reward
            
            state = next_state
            step += 1
            
        # Reconstruct discounted returns G_t
        discounted_returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + 0.95 * g
            discounted_returns.insert(0, g)
            
        for coords_t, p, g in zip(states_in_episode, mcts_policies, discounted_returns):
            episode_transitions.append((coords_t, p, g))
            
        # Add to buffer
        buffer.add_episode(episode_transitions, total_undiscounted_reward)
        
        # Optimization Step
        if len(buffer) >= 32:
            b_coords, b_mcts_probs, b_returns = buffer.sample(32)
            
            coords_tensors = torch.cat(b_coords, dim=0).to(device)
            mcts_probs_tensor = torch.tensor(np.array(b_mcts_probs), dtype=torch.float32).to(device)
            returns_tensor = torch.tensor(np.array(b_returns), dtype=torch.float32).to(device)
            
            model.train()
            policy_logits, values = model(coords_tensors)
            
            advantages = returns_tensor - values.squeeze(-1).detach()
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            
            rl_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_probs, dim=-1))
            
            # Element-wise beta
            mean_best_reward = buffer.get_mean_best_reward()
            if hasattr(model, "get_orbit_variance"):
                # Compute orbit variance (gradient-tracked)
                orbit_var = model.get_orbit_variance(coords_tensors)
                
                # Apply ablation scaling function g(Omega)
                if ablation_type == "mult":
                    # Multiplicative: g(Ω) = Ω / (Ω + λ)
                    g_omega = orbit_var.detach() / (orbit_var.detach() + lambda_val)
                    betas = torch.clamp(mean_best_reward - values.squeeze(-1), min=0.0) * alpha * g_omega
                elif ablation_type == "inv":
                    # Inverse: g(Ω) = λ / (Ω + λ)
                    g_omega = lambda_val / (orbit_var.detach() + lambda_val)
                    betas = torch.clamp(mean_best_reward - values.squeeze(-1), min=0.0) * alpha * g_omega
                else: # "baseline"
                    # Baseline: g(Ω) = 1
                    betas = torch.clamp(mean_best_reward - values.squeeze(-1), min=0.0) * alpha
            else:
                betas = torch.clamp(mean_best_reward - values.squeeze(-1), min=0.0) * alpha
            
            kl_samples = F.kl_div(log_probs, mcts_probs_tensor, reduction="none").sum(dim=-1)
            kl_loss = torch.mean(betas * kl_samples)
            value_loss = F.mse_loss(values.squeeze(-1), returns_tensor)
            
            # Add orbit consistency regularizer: eta * E_s[Omega(s)]
            if hasattr(model, "get_orbit_variance") and eta > 0.0:
                reg_loss = eta * torch.mean(orbit_var)
                total_loss = rl_loss + kl_loss + 0.5 * value_loss + reg_loss
            else:
                total_loss = rl_loss + kl_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        if episode % 50 == 0:
            buffer_avg = buffer.get_running_average()
            print(f"Episode {episode:03d} | "
                  f"Total Return: {total_undiscounted_reward:6.2f} | "
                  f"Buffer Moving Avg: {buffer_avg:6.2f} | "
                  f"Best Return Avg: {buffer.get_mean_best_reward():6.2f} | "
                  f"Buffer Size: {len(buffer)}")
            
    # Save model weights
    os.makedirs("checkpoints", exist_ok=True)
    model_name = "equivariant" if model_class == OctahedralRoboticNet else "standard"
    if model_name == "equivariant":
        save_path = f"checkpoints/{model_name}_{ablation_type}_robotic_model_{seed}.pth"
    else:
        save_path = f"checkpoints/{model_name}_robotic_model_{seed}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")
    return model, buffer.episode_returns

def run_3d_robotic_ablation(num_episodes=600):
    seeds = [42, 100, 2026, 7, 777]
    
    for seed in seeds:
        std_path = f"checkpoints/standard_robotic_model_{seed}.pth"
        eq_path = f"checkpoints/equivariant_robotic_model_{seed}.pth"
        
        # 1. Train Standard CNN AC Agent
        if not os.path.exists(std_path):
            print(f"\n=== Training 3D Standard AC Agent (seed={seed}) ===")
            train_robotic_agent(StandardRoboticAC, num_episodes=num_episodes, seed=seed)
        else:
            print(f"Checkpoint {std_path} exists. Skipping standard training.")
            
        # 2. Train SO(3) Equivariant Agent
        if not os.path.exists(eq_path):
            print(f"\n=== Training 3D SO(3) Equivariant Agent (seed={seed}) ===")
            train_robotic_agent(OctahedralRoboticNet, num_episodes=num_episodes, seed=seed)
        else:
            print(f"Checkpoint {eq_path} exists. Skipping equivariant training.")
    
    # 3. Trigger Robustness Evaluator
    print("\n=== Triggering Robustness Evaluation Suite ===")
    import robustness_eval
    robustness_eval.run_robustness_suite()

if __name__ == "__main__":
    run_3d_robotic_ablation(num_episodes=600)
