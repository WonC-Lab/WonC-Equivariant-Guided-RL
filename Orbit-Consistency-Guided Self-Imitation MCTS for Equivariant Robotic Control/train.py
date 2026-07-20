import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from self_imitation_env import SelfImitationNavigationEnv
from models import StandardCNN, D4EquivariantNet
from mcts import ActorCriticMCTS
from replay_buffer import SymmetricSelfImitationBuffer

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_agent(model_class, num_episodes=300, max_steps=40, mcts_searches=15, alpha=0.1, random_obstacles=True, seed=42, ablation_type="mult", eta=0.05, lambda_val=0.05):
    set_seed(seed)
    
    # Initialize components
    env = SelfImitationNavigationEnv(size=13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class(board_size=13, in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    mcts = ActorCriticMCTS(model, c_puct=1.5)
    buffer = SymmetricSelfImitationBuffer(max_size=10000, running_avg_window=30)
    
    # Track statistics
    history_rewards = []
    history_beta = []
    history_collisions = []
    
    total_collisions = 0
    
    print(f"Training {model_class.__name__} (episodes={num_episodes}, searches={mcts_searches}, alpha={alpha}, seed={seed})...")
    for episode in range(1, num_episodes + 1):
        if random_obstacles:
            env.randomize_obstacles()
        else:
            env.reset_canonical_obstacles()
            
        state = env.generate_initial_state()
        
        episode_transitions = []
        total_undiscounted_reward = 0.0
        
        # MCTS temperature
        temp = 1.0 if episode < 50 else 0.5
        
        step = 0
        game_over = False
        winner = None
        
        # Rollout Episode using MCTS
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
            
            full_prob = np.zeros(8)
            for act, prob in zip(actions, probs):
                full_prob[act] = prob
                
            states_in_episode.append(env.clone_state(state))
            actions_taken.append(action)
            mcts_policies.append(full_prob)
            
            next_state, _ = env.step(state, action)
            
            game_over, winner = env.check_game_over(next_state)
            if game_over:
                if winner == 1:
                    step_reward = 10.0
                else:
                    step_reward = -5.0
                    total_collisions += 1
            else:
                step_reward = -0.1
                
            rewards.append(step_reward)
            total_undiscounted_reward += step_reward
            
            state = next_state
            step += 1
            
        # Reconstruct discounted returns G_t
        discounted_returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + 0.95 * g  # inline gamma
            discounted_returns.insert(0, g)
            
        for s, p, g in zip(states_in_episode, mcts_policies, discounted_returns):
            episode_transitions.append((s, p, g))
            
        # Add episode to buffer
        buffer.add_episode(episode_transitions, total_undiscounted_reward)
        history_rewards.append(total_undiscounted_reward)
        history_collisions.append(total_collisions)
        
        # Optimization Step
        if len(buffer) >= 32:  # batch_size=32
            b_states, b_mcts_probs, b_returns = buffer.sample(32)
            
            state_tensors = torch.cat([env.state_to_tensor(s) for s in b_states], dim=0).to(device)
            mcts_probs_tensor = torch.tensor(np.array(b_mcts_probs), dtype=torch.float32).to(device)
            returns_tensor = torch.tensor(np.array(b_returns), dtype=torch.float32).to(device)
            
            model.train()
            policy_logits, values = model(state_tensors)
            
            advantages = returns_tensor - values.squeeze(-1).detach()
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            
            rl_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_probs, dim=-1))
            
            # Element-wise state-dependent beta
            mean_best_reward = buffer.get_mean_best_reward()
            if hasattr(model, "get_orbit_variance"):
                # Compute orbit variance (gradient-tracked)
                orbit_var = model.get_orbit_variance(state_tensors)
                
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
            
            history_beta.append(betas.mean().item())
        else:
            history_beta.append(0.0)
            
    # Save model weights
    os.makedirs("checkpoints", exist_ok=True)
    model_name = "equivariant" if model_class == D4EquivariantNet else "standard"
    if model_name == "equivariant":
        save_path = f"checkpoints/{model_name}_{ablation_type}_academic_model_{seed}.pth"
    else:
        save_path = f"checkpoints/{model_name}_academic_model_{seed}.pth"
    torch.save(model.state_dict(), save_path)
    
    return history_rewards, history_collisions, model

def evaluate_agent(model, env, num_eval_episodes=20, mcts_searches=15, seed=42):
    """Evaluates agent greedy performance over unseen random maps."""
    set_seed(seed)
    success_count = 0
    mcts = ActorCriticMCTS(model, c_puct=1.5)
    
    for _ in range(num_eval_episodes):
        env.randomize_obstacles()
        state = env.generate_initial_state()
        
        step = 0
        game_over = False
        winner = None
        while not game_over and step < 40:
            actions, probs = mcts.get_action_probabilities(
                state, current_turn=1, game_env=env, num_searches=mcts_searches, temp=0.0  # greedy
            )
            if not actions:
                break
            action = actions[np.argmax(probs)]
            state, _ = env.step(state, action)
            game_over, winner = env.check_game_over(state)
            step += 1
            
        if game_over and winner == 1:
            success_count += 1
            
    return success_count / num_eval_episodes

def run_academic_suite(num_episodes=300):
    env = SelfImitationNavigationEnv(size=13)
    seeds = [42, 100, 2026]
    
    print("\n=========================================")
    print("Running Comparative Study under 3 Seeds (Standard vs Equivariant)...")
    
    std_all_rewards = []
    std_all_collisions = []
    std_success_rates = []
    
    eq_all_rewards = []
    eq_all_collisions = []
    eq_success_rates = []
    
    std_last_model = None
    eq_last_model = None
    
    for seed in seeds:
        # Standard
        std_rewards, std_collisions, std_model = train_agent(StandardCNN, num_episodes=num_episodes, random_obstacles=True, seed=seed)
        std_all_rewards.append(std_rewards)
        std_all_collisions.append(std_collisions)
        std_success = evaluate_agent(std_model, env, num_eval_episodes=20, seed=seed)
        std_success_rates.append(std_success)
        std_last_model = std_model
        
        # Equivariant
        eq_rewards, eq_collisions, eq_model = train_agent(D4EquivariantNet, num_episodes=num_episodes, random_obstacles=True, seed=seed)
        eq_all_rewards.append(eq_rewards)
        eq_all_collisions.append(eq_collisions)
        eq_success = evaluate_agent(eq_model, env, num_eval_episodes=20, seed=seed)
        eq_success_rates.append(eq_success)
        eq_last_model = eq_model
        
    std_mean_success = np.mean(std_success_rates)
    std_std_success = np.std(std_success_rates)
    
    eq_mean_success = np.mean(eq_success_rates)
    eq_std_success = np.std(eq_success_rates)
    
    print(f"\nStandard CNN Success Rate: {std_mean_success:.2%} +- {std_std_success:.2%}")
    print(f"Equivariant Net Success Rate: {eq_mean_success:.2%} +- {eq_std_success:.2%}")
    
    # 2. MCTS Searches Sensitivity Sweeps (D4 Equivariant Net, Single Seed 42, 300 episodes)
    print("\n=========================================")
    print("Running MCTS Search Budget Sweeps...")
    searches_sweep = {}
    for s_budget in [5, 30]:
        rewards, _, _ = train_agent(D4EquivariantNet, num_episodes=num_episodes, mcts_searches=s_budget, random_obstacles=True, seed=42)
        searches_sweep[s_budget] = rewards
    # Add the seed 42 run for 15 searches
    searches_sweep[15] = eq_all_rewards[0]
    
    # 3. Regularization Weight Alpha Sensitivity Sweeps (D4 Equivariant Net, Single Seed 42, 300 episodes)
    print("\n=========================================")
    print("Running Alpha Scale Factor Sweeps...")
    alpha_sweep = {}
    for a_val in [0.0, 0.2]:
        rewards, _, _ = train_agent(D4EquivariantNet, num_episodes=num_episodes, alpha=a_val, random_obstacles=True, seed=42)
        alpha_sweep[a_val] = rewards
    # Add the seed 42 run for 0.1 alpha
    alpha_sweep[0.1] = eq_all_rewards[0]
    
    # Plotting 2x2 Subplots with Shaded Variance
    plot_academic_results(
        std_all_rewards, std_all_collisions, std_mean_success, std_std_success,
        eq_all_rewards, eq_all_collisions, eq_mean_success, eq_std_success,
        searches_sweep, alpha_sweep
    )

def plot_academic_results(std_all_rewards, std_all_collisions, std_mean_success, std_std_success,
                           eq_all_rewards, eq_all_collisions, eq_mean_success, eq_std_success,
                           searches_sweep, alpha_sweep):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Process return arrays
    std_rew_arr = np.array(std_all_rewards)  # (3, 300)
    eq_rew_arr = np.array(eq_all_rewards)    # (3, 300)
    
    # Process collision arrays
    std_col_arr = np.array(std_all_collisions)  # (3, 300)
    eq_col_arr = np.array(eq_all_collisions)    # (3, 300)
    
    # Moving average helper
    def moving_avg(data, window=10):
        ma = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ma[i, j] = np.mean(data[i, max(0, j-window+1):j+1])
        return ma
        
    std_rew_ma = moving_avg(std_rew_arr)
    eq_rew_ma = moving_avg(eq_rew_arr)
    
    # Subplot 1: Returns Comparison (Standard vs Equivariant)
    std_mean = np.mean(std_rew_ma, axis=0)
    std_std = np.std(std_rew_ma, axis=0)
    eq_mean = np.mean(eq_rew_ma, axis=0)
    eq_std = np.std(eq_rew_ma, axis=0)
    
    axs[0, 0].plot(std_mean, color="red", linewidth=2.5, label=f"Standard CNN ({std_mean_success:.1%} ± {std_std_success:.1%})")
    axs[0, 0].fill_between(range(len(std_mean)), std_mean - std_std, std_mean + std_std, color="red", alpha=0.15)
    
    axs[0, 0].plot(eq_mean, color="blue", linewidth=2.5, label=f"D4-Equivariant ({eq_mean_success:.1%} ± {eq_std_success:.1%})")
    axs[0, 0].fill_between(range(len(eq_mean)), eq_mean - eq_std, eq_mean + eq_std, color="blue", alpha=0.15)
    
    axs[0, 0].set_title("Training Return (Mean ± Std Dev, 3 Seeds)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Return")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Subplot 2: Cumulative Collisions
    std_col_mean = np.mean(std_col_arr, axis=0)
    std_col_std = np.std(std_col_arr, axis=0)
    eq_col_mean = np.mean(eq_col_arr, axis=0)
    eq_col_std = np.std(eq_col_arr, axis=0)
    
    axs[0, 1].plot(std_col_mean, color="red", linewidth=2.5, label="Standard CNN")
    axs[0, 1].fill_between(range(len(std_col_mean)), std_col_mean - std_col_std, std_col_mean + std_col_std, color="red", alpha=0.15)
    
    axs[0, 1].plot(eq_col_mean, color="blue", linewidth=2.5, label="D4 Equivariant Net")
    axs[0, 1].fill_between(range(len(eq_col_mean)), eq_col_mean - eq_col_std, eq_col_mean + eq_col_std, color="blue", alpha=0.15)
    
    axs[0, 1].set_title("Exploration Safety: Cumulative Collisions (3 Seeds)")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Cumulative Collisions")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Subplot 3: MCTS Searches Sensitivity
    colors = {5: "coral", 15: "blue", 30: "indigo"}
    for searches, rewards in sorted(searches_sweep.items()):
        rewards_ma = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]
        axs[1, 0].plot(rewards_ma, color=colors[searches], linewidth=2.5, label=f"Searches = {searches}")
    axs[1, 0].set_title("Sensitivity Analysis: MCTS Simulation Budget")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Moving Avg Return (10)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Subplot 4: Alpha Sensitivity
    alpha_colors = {0.0: "grey", 0.1: "blue", 0.2: "darkgreen"}
    for alpha, rewards in sorted(alpha_sweep.items()):
        rewards_ma = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]
        label_text = f"Alpha = {alpha} (Baseline)" if alpha == 0.1 else f"Alpha = {alpha}"
        if alpha == 0.0:
            label_text = "Alpha = 0.0 (No Self-Imitation)"
        axs[1, 1].plot(rewards_ma, color=alpha_colors[alpha], linewidth=2.5, label=label_text)
    axs[1, 1].set_title("Sensitivity Analysis: Regularization Weight (Alpha)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Moving Avg Return (10)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = "academic_comparison_results.png"
    plt.savefig(plot_path)
    print(f"Saved comparative academic chart to {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_academic_suite(num_episodes=300)
