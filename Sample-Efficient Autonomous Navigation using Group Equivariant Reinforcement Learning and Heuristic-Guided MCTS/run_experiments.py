import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import numpy as np
import torch
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import codebase components
from autonomous_env import AutonomousNavigationEnv
from equivariant_models import D4EquivariantNet, StandardCNN, D4GroupAction
from mcts_actor_critic import ActorCriticMCTS
from heuristic_guided_loss import HeuristicGuidedLoss
from train_navigation import SymmetricNavEnvAdapter

# Utility to transform actions for data augmentation
def transform_action(action_idx, action_i, board_size=13):
    one_hot = torch.zeros(1, 1, board_size, board_size)
    r = action_idx // board_size
    c = action_idx % board_size
    one_hot[0, 0, r, c] = 1.0
    transformed = D4GroupAction.apply_action(one_hot, action_i)
    flat_idx = torch.argmax(transformed).item()
    return flat_idx

# Utility to transform heuristic probabilities for data augmentation
def transform_probs(probs, action_i, board_size=13):
    grid = torch.tensor(probs).view(1, 1, board_size, board_size)
    transformed = D4GroupAction.apply_action(grid, action_i)
    return transformed.flatten().numpy()

# Core training function for a given configuration
def train_config(
    model_type='equivariant', # 'equivariant' or 'standard'
    use_mcts=True,
    beta_start=1.0,
    beta_decay=0.95,
    num_episodes=200,
    num_searches=15,
    max_steps=20,
    data_augmentation=False
):
    print(f"\nTraining configuration: model={model_type}, use_mcts={use_mcts}, beta_start={beta_start}, data_aug={data_augmentation}")
    
    base_env = AutonomousNavigationEnv(size=13)
    env = SymmetricNavEnvAdapter(base_env)
    
    # Initialize appropriate network with 6 layers to cover the 13x13 grid board
    if model_type == 'equivariant':
        model = D4EquivariantNet(board_size=13, in_channels=3, num_filters=16, num_layers=6)
    else:
        model = StandardCNN(board_size=13, in_channels=3, num_filters=16, num_layers=6)
        
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    if use_mcts:
        mcts = ActorCriticMCTS(model=model, c_puct=1.4)
        
    loss_fn = HeuristicGuidedLoss(beta_start=beta_start, beta_decay=beta_decay, beta_min=0.5)
    gamma = 0.95
    
    # Track statistics
    success_history = []
    cumulative_collisions = 0
    collision_history = []
    steps_history = []
    loss_history = []
    beta_history = []
    
    start_time = time.time()
    
    for ep in range(1, num_episodes + 1):
        state = base_env.generate_initial_state()
        
        episode_states = []
        episode_actions = []
        episode_heuristics = []
        episode_old_log_probs = []
        
        step = 0
        game_over = False
        winner = None
        
        # 1. Episode Simulation
        while not game_over and step < max_steps:
            state_tensor = env.state_to_tensor(state, turn=1)
            
            # Query model for old log probs before taking step
            model.eval()
            with torch.no_grad():
                logits, _ = model(state_tensor)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            if use_mcts:
                actions, probs = mcts.get_action_probabilities(
                    state, current_turn=1, game_env=env, num_searches=num_searches, temp=0.3
                )
                
                # Map probabilities back to flat 169 action space
                search_probs = np.zeros(13 * 13, dtype=np.float32)
                for act, prob in zip(actions, probs):
                    search_probs[act] = prob
                chosen_action = np.random.choice(actions, p=probs)
            else:
                # Mask valid actions
                valid_acts = env.get_valid_actions(state, turn=1)
                probs_flat = torch.softmax(logits.squeeze(0), dim=0).numpy()
                masked_probs = np.zeros(13 * 13, dtype=np.float32)
                for act in valid_acts:
                    masked_probs[act] = probs_flat[act]
                
                sum_p = np.sum(masked_probs)
                if sum_p > 0:
                    masked_probs /= sum_p
                else:
                    masked_probs[valid_acts] = 1.0 / len(valid_acts)
                
                chosen_action = np.random.choice(np.arange(13*13), p=masked_probs)
                search_probs = masked_probs
                
            heuristic_probs = env.get_heuristic_policy_flat(state)
            old_log_prob = log_probs[0, chosen_action].item()
            
            # Record state transition
            episode_states.append(state_tensor)
            episode_actions.append(chosen_action)
            episode_heuristics.append(heuristic_probs)
            episode_old_log_probs.append(old_log_prob)
            
            # Step in environment
            state, _ = env.step(state, chosen_action, turn=1)
            game_over, winner = env.check_game_over(state, turn=1)
            step += 1
            
        # 2. Episode Outcomes
        if winner == 1:
            success = 1.0
            final_reward = 1.0
        else:
            success = 0.0
            if winner == 2:
                final_reward = -1.0
                cumulative_collisions += 1
            else:
                final_reward = -0.2 # timeout
                
        success_history.append(success)
        collision_history.append(cumulative_collisions)
        steps_history.append(step)
        
        # Calculate returns
        returns = []
        g = final_reward
        for _ in reversed(episode_states):
            returns.append(g)
            g *= gamma
        returns.reverse()
        
        # 3. Model Update
        if episode_states:
            model.train()
            
            batch_states = torch.cat(episode_states, dim=0)
            batch_actions = torch.tensor(episode_actions, dtype=torch.long)
            batch_returns = torch.tensor(returns, dtype=torch.float32)
            batch_heuristics = torch.tensor(np.array(episode_heuristics), dtype=torch.float32)
            batch_old_log_probs = torch.tensor(episode_old_log_probs, dtype=torch.float32)
            
            # 8x Data Augmentation if requested (for Standard CNN baseline)
            if data_augmentation:
                augmented_states = []
                augmented_actions = []
                augmented_returns = []
                augmented_heuristics = []
                augmented_old_log_probs = []
                
                for idx in range(len(episode_states)):
                    st = episode_states[idx] # (1, 3, 13, 13)
                    act = episode_actions[idx]
                    ret = returns[idx]
                    heur = episode_heuristics[idx]
                    old_lp = episode_old_log_probs[idx]
                    
                    for i in range(8):
                        # Apply symmetry to states
                        aug_st = D4GroupAction.apply_action(st, i)
                        # Apply symmetry to actions
                        aug_act = transform_action(act, i)
                        # Apply symmetry to heuristic target probabilities
                        aug_heur = transform_probs(heur, i)
                        
                        augmented_states.append(aug_st)
                        augmented_actions.append(aug_act)
                        augmented_returns.append(ret)
                        augmented_heuristics.append(aug_heur)
                        augmented_old_log_probs.append(old_lp)
                        
                batch_states = torch.cat(augmented_states, dim=0)
                batch_actions = torch.tensor(augmented_actions, dtype=torch.long)
                batch_returns = torch.tensor(augmented_returns, dtype=torch.float32)
                batch_heuristics = torch.tensor(np.array(augmented_heuristics), dtype=torch.float32)
                batch_old_log_probs = torch.tensor(augmented_old_log_probs, dtype=torch.float32)
            
            avg_loss = 0.0
            for _ in range(5):
                optimizer.zero_grad()
                logits, values = model(batch_states)
                loss, rl_loss_val, kl_loss_val, val_loss_val = loss_fn(
                    logits, values, batch_actions, batch_returns, batch_heuristics, batch_old_log_probs
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= 5.0
            loss_fn.decay_beta()
            
            loss_history.append(avg_loss)
            beta_history.append(loss_fn.beta)
        else:
            loss_history.append(0.0)
            beta_history.append(loss_fn.beta)
            
        if ep % 10 == 0:
            avg_succ = np.mean(success_history[-10:]) * 100
            print(f"Episode {ep:02d}/{num_episodes} | Avg Success (last 10): {avg_succ:.1f}% | Collisions: {cumulative_collisions} | Beta: {loss_fn.beta:.3f}")
            
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f} seconds. Success Rate: {np.mean(success_history[-10:])*100:.1f}%")
    
    return {
        'model': model,
        'success_history': success_history,
        'collision_history': collision_history,
        'steps_history': steps_history,
        'loss_history': loss_history,
        'beta_history': beta_history,
        'training_time': elapsed
    }

# Evaluates generalization by rotating environment observations and realigning predictions
def evaluate_generalization(model, num_trials=10, use_mcts=True, num_searches=15):
    base_env = AutonomousNavigationEnv(size=13)
    env = SymmetricNavEnvAdapter(base_env)
    
    if use_mcts:
        mcts = ActorCriticMCTS(model=model, c_puct=1.4)
        
    results_per_action = {}
    
    for i in range(8):
        success_count = 0
        for _ in range(num_trials):
            state = base_env.generate_initial_state()
            step = 0
            game_over = False
            winner = None
            
            while not game_over and step < 20:
                if use_mcts:
                    # To test group actions: we apply action i to the MCTS selection states
                    # A clean way is to intercept during selection/expansion inside MCTS.
                    # Or, we can run standard MCTS, but we rotate the observation at the model query.
                    # In mcts_actor_critic.py, the model is queried inside run_simulation:
                    # `state_tensor = game_env.state_to_tensor(state, turn)`
                    # We can simulate this by wrapping the model itself with a rotated wrapper!
                    
                    class RotatedModelWrapper(torch.nn.Module):
                        def __init__(self, base_model, action_idx):
                            super().__init__()
                            self.base_model = base_model
                            self.action_idx = action_idx
                        def forward(self, x):
                            # Rotate input x
                            gx = D4GroupAction.apply_action(x, self.action_idx)
                            logits, value = self.base_model(gx)
                            
                            # Realign logits using inverse action
                            logits_grid = logits.reshape(-1, 1, 13, 13)
                            inv_logits_grid = D4GroupAction.apply_inverse_action(logits_grid, self.action_idx)
                            inv_logits = inv_logits_grid.reshape(-1, 13 * 13)
                            return inv_logits, value
                    
                    wrapped_model = RotatedModelWrapper(model, i)
                    wrapped_mcts = ActorCriticMCTS(model=wrapped_model, c_puct=1.4)
                    
                    actions, probs = wrapped_mcts.get_action_probabilities(
                        state, current_turn=1, game_env=env, num_searches=num_searches, temp=0.0
                    )
                    chosen_action = actions[np.argmax(probs)]
                else:
                    # Direct query
                    model.eval()
                    x = env.state_to_tensor(state, turn=1)
                    gx = D4GroupAction.apply_action(x, i)
                    with torch.no_grad():
                        logits, _ = model(gx)
                    logits_grid = logits.reshape(1, 1, 13, 13)
                    inv_logits_grid = D4GroupAction.apply_inverse_action(logits_grid, i)
                    inv_logits = inv_logits_grid.reshape(1, 13*13)
                    
                    valid_acts = env.get_valid_actions(state, turn=1)
                    probs_flat = torch.softmax(inv_logits.squeeze(0), dim=0).numpy()
                    masked_probs = np.zeros(13 * 13, dtype=np.float32)
                    for act in valid_acts:
                        masked_probs[act] = probs_flat[act]
                    chosen_action = np.argmax(masked_probs)
                    
                state, _ = env.step(state, chosen_action, turn=1)
                game_over, winner = env.check_game_over(state, turn=1)
                step += 1
                
            if winner == 1:
                success_count += 1
                
        results_per_action[i] = success_count / num_trials * 100
        
    return results_per_action


def main():
    print("=============================================================")
    print(" Running the 5-Experiment Evaluation Suite for RL & MCTS...")
    print("=============================================================")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: Ablation Study
    # -----------------------------------------------------------------
    print("\n--- Running Experiment 1: Ablation Study ---")
    
    # 1. Full (Proposed)
    exp1_full = train_config(model_type='equivariant', use_mcts=True, beta_start=1.0)
    
    # 2. w/o Equivariance (Standard CNN + MCTS + Guidance)
    exp1_no_equi = train_config(model_type='standard', use_mcts=True, beta_start=1.0)
    
    # 3. w/o Heuristic Guidance (Equivariant + MCTS, beta=0)
    exp1_no_guide = train_config(model_type='equivariant', use_mcts=True, beta_start=0.0)
    
    # 4. w/o MCTS (Equivariant + Guidance, direct policy)
    exp1_no_mcts = train_config(model_type='equivariant', use_mcts=False, beta_start=1.0)
    
    # Plot Experiment 1: Ablation Study
    plt.figure(figsize=(8, 5))
    
    def smooth(y, box_pts=5):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        # fix edges
        for idx in range(box_pts):
            y_smooth[idx] = np.mean(y[:idx+1])
            y_smooth[-idx-1] = np.mean(y[-idx-1:])
        return y_smooth

    plt.plot(smooth(exp1_full['success_history']), label='Full Framework (Proposed)', color='#1f77b4', linewidth=2.5)
    plt.plot(smooth(exp1_no_equi['success_history']), label='w/o Equivariance (Std CNN)', color='#ff7f0e', linestyle='--')
    plt.plot(smooth(exp1_no_guide['success_history']), label='w/o Heuristic Guidance', color='#2ca02c', linestyle=':')
    plt.plot(smooth(exp1_no_mcts['success_history']), label='w/o MCTS (Direct Policy)', color='#d62728', linestyle='-.')
    
    plt.title('Ablation Study: Convergence of Framework Components', fontsize=12, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Success Rate (Moving Avg)', fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # EXPERIMENT 2: Zero-shot Generalization
    # -----------------------------------------------------------------
    print("\n--- Running Experiment 2: Zero-shot Generalization ---")
    
    # We need to train Standard CNN with 8x Data Augmentation for comparison
    exp2_std_aug = train_config(model_type='standard', use_mcts=True, beta_start=1.0, data_augmentation=True)
    
    # Evaluate models on all 8 dihedral transformations
    print("Evaluating generalization on D4 actions (10 trials each)...")
    gen_equivariant = evaluate_generalization(exp1_full['model'])
    gen_standard = evaluate_generalization(exp1_no_equi['model'])
    gen_std_aug = evaluate_generalization(exp2_std_aug['model'])
    
    # Plot Experiment 2: Zero-shot Generalization
    plt.figure(figsize=(9, 5))
    x = np.arange(8)
    width = 0.25
    
    plt.bar(x - width, [gen_equivariant[i] for i in range(8)], width, label='D4-Net (Symmetric, Ours)', color='#1f77b4')
    plt.bar(x, [gen_std_aug[i] for i in range(8)], width, label='Standard CNN (8x Augmented)', color='#ff7f0e')
    plt.bar(x + width, [gen_standard[i] for i in range(8)], width, label='Standard CNN (No Augmentation)', color='#d62728')
    
    group_labels = [f"$g_{i}$" for i in range(8)]
    plt.xticks(x, group_labels)
    plt.title('Zero-Shot Generalization Under D4 Group Actions', fontsize=12, fontweight='bold')
    plt.xlabel('Dihedral Group Transformation Action Index', fontsize=10)
    plt.ylabel('Test Success Rate (%)', fontsize=10)
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig('generalization_test.png', dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # EXPERIMENT 3: Safety Analysis in Early Stage
    # -----------------------------------------------------------------
    print("\n--- Running Experiment 3: Safety Analysis ---")
    
    plt.figure(figsize=(8, 5))
    plt.plot(exp1_full['collision_history'], label='Heuristic-Guided RL (Beta=1.0)', color='#1f77b4', linewidth=2.5)
    plt.plot(exp1_no_guide['collision_history'], label='Pure RL Exploration (Beta=0)', color='#d62728', linestyle='--', linewidth=2.0)
    
    plt.title('Safety Analysis: Cumulative Collisions During Training', fontsize=12, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Cumulative Obstacle Collisions', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig('exploration_safety.png', dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # EXPERIMENT 4: Sample Efficiency Curves
    # -----------------------------------------------------------------
    print("\n--- Running Experiment 4: Sample Efficiency ---")
    
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(exp1_full['success_history']), label='D4-Net + MCTS + Guidance (Proposed)', color='#1f77b4', linewidth=2.5)
    plt.plot(smooth(exp2_std_aug['success_history']), label='Standard CNN + Augmentation (8x data)', color='#ff7f0e', linewidth=1.5, linestyle='--')
    plt.plot(smooth(exp1_no_equi['success_history']), label='Standard CNN (No Augmentation)', color='#d62728', linewidth=1.5, linestyle=':')
    
    plt.title('Sample Efficiency & Training Convergence Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Success Rate (Moving Avg)', fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig('sample_efficiency.png', dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # EXPERIMENT 5: Sensitivity Analysis of MCTS Searches
    # -----------------------------------------------------------------
    print("\n--- Running Experiment 5: MCTS Sensitivity Analysis ---")
    
    # We will test N_search = 5, 15, and 30
    print("Testing MCTS Search Count: N=5")
    exp5_n5 = train_config(model_type='equivariant', use_mcts=True, beta_start=1.0, num_searches=5)
    
    print("Testing MCTS Search Count: N=30")
    exp5_n30 = train_config(model_type='equivariant', use_mcts=True, beta_start=1.0, num_searches=30)
    
    # Plot 5a: Learning Curves for different N
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(exp5_n5['success_history']), label='MCTS Simulations N=5', color='#2ca02c', linestyle=':')
    plt.plot(smooth(exp1_full['success_history']), label='MCTS Simulations N=15', color='#1f77b4', linewidth=2.0)
    plt.plot(smooth(exp5_n30['success_history']), label='MCTS Simulations N=30', color='#9467bd', linestyle='--')
    
    plt.title('MCTS Scale Sensitivity: Learning Curve comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Success Rate (Moving Avg)', fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig('mcts_sensitivity.png', dpi=300)
    plt.close()
    
    # Create quantitative data report JSON
    report = {
        'ablation': {
            'full_success_rate': float(np.mean(exp1_full['success_history'][-10:]) * 100),
            'no_equi_success_rate': float(np.mean(exp1_no_equi['success_history'][-10:]) * 100),
            'no_guide_success_rate': float(np.mean(exp1_no_guide['success_history'][-10:]) * 100),
            'no_mcts_success_rate': float(np.mean(exp1_no_mcts['success_history'][-10:]) * 100),
        },
        'generalization': {
            'equivariant': [float(gen_equivariant[i]) for i in range(8)],
            'std_aug': [float(gen_std_aug[i]) for i in range(8)],
            'standard': [float(gen_standard[i]) for i in range(8)]
        },
        'safety': {
            'guided_collisions': int(exp1_full['collision_history'][-1]),
            'unguided_collisions': int(exp1_no_guide['collision_history'][-1])
        },
        'sample_efficiency': {
            'equivariant_episodes': 40,
            'equivariant_success': float(np.mean(exp1_full['success_history'][-10:]) * 100),
            'std_aug_success': float(np.mean(exp2_std_aug['success_history'][-10:]) * 100),
            'std_success': float(np.mean(exp1_no_equi['success_history'][-10:]) * 100)
        },
        'sensitivity': {
            'n5_time_per_ep': float(exp5_n5['training_time'] / 40),
            'n15_time_per_ep': float(exp1_full['training_time'] / 40),
            'n30_time_per_ep': float(exp5_n30['training_time'] / 40),
            'n5_success': float(np.mean(exp5_n5['success_history'][-10:]) * 100),
            'n15_success': float(np.mean(exp1_full['success_history'][-10:]) * 100),
            'n30_success': float(np.mean(exp5_n30['success_history'][-10:]) * 100),
        }
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n=============================================================")
    print(" All experiments successfully executed! Plots saved.")
    print(" Quantitative data written to 'experiment_results.json'.")
    print("=============================================================")


if __name__ == "__main__":
    main()
