import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from multi_agent_env import MultiAgentNavigationEnv
from equivariant_gnn import PermutationEquivariantGNN
from multi_agent_mcts import MultiAgentMCTS

def evaluate_performance(model, num_agents, obstacle_mode="default", num_episodes=10, mode="mcts", mcts_searches=40):
    size = 13
    env = MultiAgentNavigationEnv(size=size, num_agents=num_agents)
    
    if obstacle_mode == "empty":
        env.obstacles = set()
    elif obstacle_mode == "random":
        import random
        random.seed(42) # Reproducible random obstacles
        env.obstacles = set()
        starts_goals = set(env.default_starts + env.default_goals)
        while len(env.obstacles) < 12:
            r = random.randint(1, 11)
            c = random.randint(1, 11)
            if (r, c) not in starts_goals:
                env.obstacles.add((r, c))
                
    mcts = MultiAgentMCTS(model=model, c_puct=1.4)
    
    success_count = 0
    total_steps = 0
    crashed_count = 0
    
    for ep in range(num_episodes):
        state = env.generate_initial_state()
        done = False
        step = 0
        max_steps = 30
        
        while not done and step < max_steps:
            joint_action = []
            active_mask = state[3]
            
            if mode == "mcts":
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    # Search
                    _, mcts_probs = mcts.get_action_probabilities(
                        state, agent_idx=i, env=env, num_searches=mcts_searches, temp=0.0 # Greedy action
                    )
                    joint_action.append(np.argmax(mcts_probs))
            elif mode == "gnn":
                obs_joint = env.get_joint_observation(state)
                model.eval()
                with torch.no_grad():
                    logits, _ = model(obs_joint.unsqueeze(0))
                logits = logits.squeeze(0) # (M, 8)
                
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    v_actions = env.get_valid_actions(state, i)
                    a_logits = logits[i].clone()
                    inv_actions = [a for a in range(8) if a not in v_actions]
                    a_logits[inv_actions] = -1e9
                    joint_action.append(torch.argmax(a_logits).item())
            elif mode == "heuristic":
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    heur_probs = env.get_heuristic_policy(state, i)
                    v_actions = env.get_valid_actions(state, i)
                    masked_heur = np.zeros(8)
                    for a in v_actions:
                        masked_heur[a] = heur_probs[a]
                    sum_h = np.sum(masked_heur)
                    if sum_h > 0:
                        masked_heur /= sum_h
                        joint_action.append(np.argmax(masked_heur))
                    else:
                        joint_action.append(0)
                    
            state, _, done, _ = env.step(state, tuple(joint_action))
            step += 1
            
        # Stats
        end_pos, end_goal, _, _ = state
        reached = sum([1 for i in range(num_agents) if end_pos[i] == end_goal[i]])
        if reached == num_agents:
            success_count += 1
        else:
            crashed_count += 1
        total_steps += step
        
    avg_steps = total_steps / num_episodes
    success_rate = success_count / num_episodes
    return success_rate, avg_steps

def run_experiments():
    print("=============================================================")
    print(" Running Scalability, Ablation & Robustness Experiments...")
    print("=============================================================")
    
    model = PermutationEquivariantGNN(grid_size=13, in_channels=3, d_model=128)
    
    model_path = "models/multi_agent_model.pth"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found! Please run train_multi_agent.py first.")
        print("Using randomly initialized model weights for evaluation...")
    else:
        model.load_state_dict(torch.load(model_path))
        print("Loaded trained model weights successfully.")
        
    agent_counts = [2, 3, 4, 5, 6, 8]
    obstacle_modes = ["default", "empty", "random"]
    
    results = {}
    
    # 1. Evaluate GNN+MCTS across different obstacle layouts to test robustness
    for obs_mode in obstacle_modes:
        results[obs_mode] = {"rates": [], "steps": []}
        print(f"\n--- Evaluating Robustness: Obstacle Mode = {obs_mode.upper()} ---")
        for m in agent_counts:
            print(f"PE-GNN+MCTS | M = {m} agents...")
            sr, st = evaluate_performance(model, num_agents=m, obstacle_mode=obs_mode, num_episodes=10, mode="mcts")
            results[obs_mode]["rates"].append(sr)
            results[obs_mode]["steps"].append(st)
            print(f"  Success Rate: {sr * 100:.1f}%, Avg Steps: {st:.2f}")

    # 2. Evaluate baselines on the Default Map for ablation comparison
    baseline_results = {
        "gnn": {"rates": [], "steps": []},
        "heuristic": {"rates": [], "steps": []}
    }
    
    print("\n--- Evaluating Baselines (Default Map) ---")
    for mode in ["gnn", "heuristic"]:
        for m in agent_counts:
            print(f"Baseline: {mode.upper()} | M = {m} agents...")
            sr, st = evaluate_performance(model, num_agents=m, obstacle_mode="default", num_episodes=10, mode=mode)
            baseline_results[mode]["rates"].append(sr)
            baseline_results[mode]["steps"].append(st)
            print(f"  Success Rate: {sr * 100:.1f}%, Avg Steps: {st:.2f}")

    # Plot results
    os.makedirs("results", exist_ok=True)
    
    # Plot 1: Robustness of GNN+MCTS across Obstacle Layouts
    plt.figure(figsize=(9, 5))
    plt.plot(agent_counts, [sr * 100 for sr in results["default"]["rates"]], 'o-', label='Default Map (Trained Layout)', linewidth=2.5, color='#1f77b4')
    plt.plot(agent_counts, [sr * 100 for sr in results["empty"]["rates"]], 's--', label='Empty Map (No Obstacles)', linewidth=2.0, color='#2ca02c')
    plt.plot(agent_counts, [sr * 100 for sr in results["random"]["rates"]], 'd-.', label='Random Obstacles (Unseen Layout)', linewidth=2.0, color='#d62728')
    plt.axvline(x=4, color='red', linestyle=':', label='Training Agent Limit (M=4)')
    plt.title('GNN+MCTS Robustness across Obstacle Configurations', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Number of Cooperative Agents', fontsize=11)
    plt.ylabel('Complete Success Rate (%)', fontsize=11)
    plt.xticks(agent_counts)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/scalability_test.png', dpi=300)
    plt.close()
    
    # Plot 2: MCTS Ablation (Comparison against baselines on Default Map)
    plt.figure(figsize=(9, 5))
    plt.plot(agent_counts, [sr * 100 for sr in results["default"]["rates"]], 'o-', label='PE-GNN + MCTS (Ours)', linewidth=2.5, color='#1f77b4')
    plt.plot(agent_counts, [sr * 100 for sr in baseline_results["gnn"]["rates"]], 's--', label='PE-GNN Only (No Search)', linewidth=2.0, color='#ff7f0e')
    plt.plot(agent_counts, [sr * 100 for sr in baseline_results["heuristic"]["rates"]], 'd-.', label='Coordinated Heuristic Only', linewidth=2.0, color='#2ca02c')
    plt.axvline(x=4, color='red', linestyle=':', label='Training Agent Limit (M=4)')
    plt.title('Ablation: Success Rate Comparison against Baselines', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Number of Cooperative Agents', fontsize=11)
    plt.ylabel('Complete Success Rate (%)', fontsize=11)
    plt.xticks(agent_counts)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/mcts_ablation.png', dpi=300)
    plt.close()

    print("=============================================================")
    print(" Experiments Completed. Graphical Results Saved to:")
    print(" - results/scalability_test.png (Robustness curves)")
    print(" - results/mcts_ablation.png (Ablation curves)")
    print("=============================================================")

if __name__ == "__main__":
    run_experiments()
