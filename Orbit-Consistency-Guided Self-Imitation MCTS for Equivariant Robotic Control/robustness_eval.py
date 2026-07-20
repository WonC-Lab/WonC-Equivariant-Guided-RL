import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

from robotic_mcts_env import RoboticMCTSEnv
from models_3d import StandardRoboticAC, OctahedralRoboticNet
from mcts import ActorCriticMCTS

def evaluate_robustness(model, noise_std=0.0, length_drift=1.0, num_episodes=5, mcts_searches=15):
    """
    Evaluates a trained model under physical and sensory perturbations:
    - noise_std: Standard deviation of Gaussian noise added to spatial coordinate observations.
    - length_drift: Multiplier applied to actual physical robotic arm link lengths (kinematic mismatch).
    """
    env = RoboticMCTSEnv(l1=1.0 * length_drift, l2=1.0 * length_drift, l3=1.0 * length_drift, max_steps=100)
    mcts = ActorCriticMCTS(model, c_puct=1.5)
    
    success_count = 0
    total_steps = 0
    collision_count = 0
    final_dists = []
    
    for _ in range(num_episodes):
        state = env.randomize_obstacles()
        
        step = 0
        game_over = False
        winner = None
        
        while not game_over and step < 100:
            noise_state = env.clone_state(state)
            if noise_std > 0.0:
                noise_state["target"] = (np.array(state["target"]) + np.random.normal(0, noise_std, 3)).tolist()
                noise_state["obstacle"] = (np.array(state["obstacle"]) + np.random.normal(0, noise_std, 3)).tolist()
            
            actions, probs = mcts.get_action_probabilities(
                noise_state, current_turn=1, game_env=env, num_searches=mcts_searches, temp=0.0  # greedy
            )
            if not actions:
                break
                
            action = actions[np.argmax(probs)]
            state, _ = env.step(state, action)
            game_over, winner = env.check_game_over(state)
            
            if game_over and winner == 2 and state["steps"] < env.max_steps:
                collision_count += 1
                
            step += 1
            
        if game_over and winner == 1:
            success_count += 1
            total_steps += step
        else:
            total_steps += env.max_steps
            
        # Final distance to target
        ee_pos = env.forward_kinematics(state["theta"])
        target_pos = np.array(state["target"])
        final_dists.append(np.linalg.norm(ee_pos - target_pos))
            
    success_rate = success_count / num_episodes
    avg_steps = total_steps / num_episodes
    avg_final_dist = np.mean(final_dists)
    
    return success_rate, avg_steps, collision_count, avg_final_dist

def run_robustness_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 100, 2026, 7, 777]
    
    noise_levels = [0.0, 0.04, 0.08, 0.12]
    drift_levels = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
    
    # Track metrics across seeds
    std_noise_results = {n: [] for n in noise_levels}
    eq_noise_results = {n: [] for n in noise_levels}
    
    std_drift_results = {d: [] for d in drift_levels}
    eq_drift_results = {d: [] for d in drift_levels}
    
    for seed in seeds:
        print(f"\n--- Evaluating robustness for Seed {seed} ---")
        std_model = StandardRoboticAC().to(device)
        eq_model = OctahedralRoboticNet().to(device)
        
        std_path = f"checkpoints/standard_robotic_model_{seed}.pth"
        eq_path = f"checkpoints/equivariant_robotic_model_{seed}.pth"
        
        if not os.path.exists(std_path) or not os.path.exists(eq_path):
            print(f"Error: Weights for seed {seed} not found.")
            continue
            
        std_model.load_state_dict(torch.load(std_path, map_location=device))
        eq_model.load_state_dict(torch.load(eq_path, map_location=device))
        std_model.eval()
        eq_model.eval()
        
        # Noise sweep
        for noise in noise_levels:
            _, _, _, std_dist = evaluate_robustness(std_model, noise_std=noise)
            _, _, _, eq_dist = evaluate_robustness(eq_model, noise_std=noise)
            std_noise_results[noise].append(std_dist)
            eq_noise_results[noise].append(eq_dist)
            
        # Drift sweep
        for drift in drift_levels:
            _, _, _, std_dist = evaluate_robustness(std_model, length_drift=drift)
            _, _, _, eq_dist = evaluate_robustness(eq_model, length_drift=drift)
            std_drift_results[drift].append(std_dist)
            eq_drift_results[drift].append(eq_dist)
            
    # Print formatted output for paper tables
    print("\n=========================================")
    print("3D Sensory Noise Average Distance Table Data:")
    print("Noise Level | Standard Dist | Equivariant Dist")
    for n in noise_levels:
        std_mean, std_std = np.mean(std_noise_results[n]), np.std(std_noise_results[n])
        eq_mean, eq_std = np.mean(eq_noise_results[n]), np.std(eq_noise_results[n])
        print(f"{n:.2f}m | {std_mean:.3f}m +- {std_std:.3f}m | {eq_mean:.3f}m +- {eq_std:.3f}m")
        
    print("\n3D Kinematic Drift Average Distance Table Data:")
    print("Drift Scale | Standard Dist | Equivariant Dist")
    for d in drift_levels:
        std_mean, std_std = np.mean(std_drift_results[d]), np.std(std_drift_results[d])
        eq_mean, eq_std = np.mean(eq_drift_results[d]), np.std(eq_drift_results[d])
        print(f"{d:.2f} | {std_mean:.3f}m +- {std_std:.3f}m | {eq_mean:.3f}m +- {eq_std:.3f}m")
    print("=========================================")
    
    # Plotting robustness with shaded variance
    plot_robustness_results(noise_levels, std_noise_results, eq_noise_results,
                            drift_levels, std_drift_results, eq_drift_results)

def plot_robustness_results(noise_levels, std_noise_dict, eq_noise_dict,
                             drift_levels, std_drift_dict, eq_drift_dict):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parse Noise dicts
    std_n_mean = [np.mean(std_noise_dict[n]) for n in noise_levels]
    std_n_std = [np.std(std_noise_dict[n]) for n in noise_levels]
    eq_n_mean = [np.mean(eq_noise_dict[n]) for n in noise_levels]
    eq_n_std = [np.std(eq_noise_dict[n]) for n in noise_levels]
    
    # Plot Noise
    axs[0].plot(noise_levels, std_n_mean, 'o-', color="crimson", linewidth=2.5, label="Standard CNN AC")
    axs[0].fill_between(noise_levels, np.array(std_n_mean) - np.array(std_n_std), np.array(std_n_mean) + np.array(std_n_std), color="crimson", alpha=0.15)
    
    axs[0].plot(noise_levels, eq_n_mean, 's-', color="royalblue", linewidth=2.5, label="Octahedral Equivariant Net")
    axs[0].fill_between(noise_levels, np.array(eq_n_mean) - np.array(eq_n_std), np.array(eq_n_mean) + np.array(eq_n_std), color="royalblue", alpha=0.15)
    
    axs[0].set_title("Robustness to Sensory Coordinate Noise (5 Seeds)")
    axs[0].set_xlabel("Sensory Noise Std Dev (meters)")
    axs[0].set_ylabel("Average Distance to Target (meters)")
    axs[0].legend()
    axs[0].grid(True)
    
    # Parse Drift dicts
    std_d_mean = [np.mean(std_drift_dict[d]) for d in drift_levels]
    std_d_std = [np.std(std_drift_dict[d]) for d in drift_levels]
    eq_d_mean = [np.mean(eq_drift_dict[d]) for d in drift_levels]
    eq_d_std = [np.std(eq_drift_dict[d]) for d in drift_levels]
    
    # Plot Drift
    axs[1].plot(drift_levels, std_d_mean, 'o-', color="crimson", linewidth=2.5, label="Standard CNN AC")
    axs[1].fill_between(drift_levels, np.array(std_d_mean) - np.array(std_d_std), np.array(std_d_mean) + np.array(std_d_std), color="crimson", alpha=0.15)
    
    axs[1].plot(drift_levels, eq_d_mean, 's-', color="royalblue", linewidth=2.5, label="Octahedral Equivariant Net")
    axs[1].fill_between(drift_levels, np.array(eq_d_mean) - np.array(eq_d_std), np.array(eq_d_mean) + np.array(eq_d_std), color="royalblue", alpha=0.15)
    
    axs[1].set_title("Robustness to Kinematic Length Drift (Sim-to-Sim, 5 Seeds)")
    axs[1].set_xlabel("Physical Link Length Scale Factor")
    axs[1].set_ylabel("Average Distance to Target (meters)")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plot_path = "robustness_comparison.png"
    plt.savefig(plot_path)
    print(f"Saved robustness comparisons chart to {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_robustness_suite()
