import numpy as np
import torch
import torch.optim as optim
import os

from multi_agent_env import MultiAgentNavigationEnv
from equivariant_gnn import PermutationEquivariantGNN
from multi_agent_mcts import MultiAgentMCTS
from heuristic_guided_loss import MultiAgentHeuristicGuidedLoss

def calculate_discounted_returns(rewards_history, active_masks_history, gamma=0.95):
    """
    Calculates individual discounted returns G_t^i for each agent.
    rewards_history: List of arrays of shape (M,)
    active_masks_history: List of arrays of shape (M,)
    """
    T = len(rewards_history)
    M = rewards_history[0].shape[0]
    returns = np.zeros((T, M), dtype=np.float32)
    
    # Run backpropagation of rewards per agent
    for i in range(M):
        g = 0.0
        # Iterate backwards
        for t in reversed(range(T)):
            # Only accumulate rewards if the agent was active at time t
            if active_masks_history[t][i] > 0.5:
                g = rewards_history[t][i] + gamma * g
            else:
                g = 0.0 # reset if inactive/crashed/finished
            returns[t, i] = g
            
    return returns

def train_multi_agent():
    print("=============================================================")
    print(" Starting Decentralized Multi-Agent Training via PE-MCTS...")
    print("=============================================================")

    # Hyperparameters
    size = 13
    num_agents = 4
    num_episodes = 250
    max_steps_per_episode = 30
    gamma = 0.95
    mcts_searches = 15
    
    # 1. Initialize environment, model, optimizer, loss
    env = MultiAgentNavigationEnv(size=size, num_agents=num_agents)
    model = PermutationEquivariantGNN(grid_size=size, in_channels=3, d_model=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    mcts = MultiAgentMCTS(model=model, c_puct=1.4)
    loss_fn = MultiAgentHeuristicGuidedLoss(beta_start=1.0, beta_decay=0.99, beta_min=0.3, clip_eps=0.2)

    success_counts = 0
    total_episodes_evaluated = 0
    recent_successes = []
    best_success_rate = 0.0

    for ep in range(1, num_episodes + 1):
        # Randomize obstacles per episode during training to enhance generalization
        env.obstacles = set()
        starts_goals = set(env.default_starts + env.default_goals)
        while len(env.obstacles) < 12:
            r = np.random.randint(1, 11)
            c = np.random.randint(1, 11)
            if (r, c) not in starts_goals:
                env.obstacles.add((r, c))

        state = env.generate_initial_state()
        
        # History buffers for the episode
        states_history = []
        actions_history = []
        old_log_probs_history = []
        heuristics_history = []
        active_masks_history = []
        rewards_history = []
        
        step = 0
        done = False
        
        while not done and step < max_steps_per_episode:
            # Current agent observations
            obs_joint = env.get_joint_observation(state) # (M, 3, H, W)
            
            # Predict joint prior policy and log_probs
            model.eval()
            with torch.no_grad():
                logits, _ = model(obs_joint.unsqueeze(0)) # (1, M, 8)
            logits = logits.squeeze(0) # (M, 8)
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            
            # Action selection per agent using independent MCTS
            joint_action = []
            joint_mcts_probs = []
            old_log_probs = []
            
            active_mask = state[3]
            
            for i in range(num_agents):
                if not active_mask[i]:
                    joint_action.append(0) # dummy action
                    old_log_probs.append(0.0)
                    joint_mcts_probs.append(np.ones(8) / 8.0)
                    continue
                    
                _, mcts_probs = mcts.get_action_probabilities(
                    state, agent_idx=i, env=env, num_searches=mcts_searches, temp=1.0, beta=loss_fn.beta
                )
                
                # Sample action
                chosen_act = np.random.choice(8, p=mcts_probs)
                
                joint_action.append(chosen_act)
                old_log_probs.append(log_probs[i, chosen_act])
                joint_mcts_probs.append(mcts_probs)
                
            # Collect coordinated heuristics for KL loss guidance
            heuristic_probs = env.get_joint_heuristic_policies(state) # (M, 8)
            
            # Record transition
            states_history.append(obs_joint.numpy())
            actions_history.append(np.array(joint_action))
            old_log_probs_history.append(np.array(old_log_probs))
            heuristics_history.append(heuristic_probs)
            active_masks_history.append(np.array([1.0 if m else 0.0 for m in active_mask]))
            
            # Take environment step
            next_state, rewards, done, _ = env.step(state, tuple(joint_action))
            rewards_history.append(np.array(rewards))
            
            state = next_state
            step += 1

        # Epilogue of episode: calculate rewards and returns
        returns = calculate_discounted_returns(rewards_history, active_masks_history, gamma=gamma)
        
        # Determine stats (how many agents reached their goal safely)
        end_agent_pos, end_goal_pos, _, active_mask = state
        reached_goals = 0
        crashed_agents = 0
        for i in range(num_agents):
            if end_agent_pos[i] == end_goal_pos[i]:
                reached_goals += 1
            else:
                crashed_agents += 1
                
        if reached_goals == num_agents:
            success_counts += 1
            result_str = f"ALL SUCCESS ({reached_goals}/{num_agents})"
            recent_successes.append(1)
        else:
            result_str = f"PARTIAL (Goal: {reached_goals}, Crash/Timeout: {crashed_agents})"
            recent_successes.append(0)
            
        if len(recent_successes) > 30:
            recent_successes.pop(0)
            
        current_success_rate = sum(recent_successes) / len(recent_successes)
        if len(recent_successes) >= 20 and current_success_rate > best_success_rate:
            best_success_rate = current_success_rate
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/multi_agent_model.pth")
            print(f"--> Saved NEW BEST model checkpoint (Success Rate: {best_success_rate*100:.1f}%)")

        # 2. Optimization step
        if states_history:
            model.train()
            
            # Convert buffers to tensors
            batch_states = torch.tensor(np.stack(states_history, axis=0), dtype=torch.float32) # (T, M, 3, H, W)
            batch_actions = torch.tensor(np.stack(actions_history, axis=0), dtype=torch.long) # (T, M)
            batch_old_log_probs = torch.tensor(np.stack(old_log_probs_history, axis=0), dtype=torch.float32) # (T, M)
            batch_returns = torch.tensor(returns, dtype=torch.float32) # (T, M)
            batch_heuristics = torch.tensor(np.stack(heuristics_history, axis=0), dtype=torch.float32) # (T, M, 8)
            batch_masks = torch.tensor(np.stack(active_masks_history, axis=0), dtype=torch.float32) # (T, M)
            
            # 4 epochs of optimization per episode batch (standard in PPO)
            last_loss = 0.0
            for _ in range(4):
                optimizer.zero_grad()
                logits, values = model(batch_states) # logits: (T, M, 8), values: (T, M, 1)
                
                loss, p_loss, kl_loss, v_loss = loss_fn(
                    policy_logits=logits,
                    values=values,
                    actions=batch_actions,
                    old_log_probs=batch_old_log_probs,
                    rl_returns=batch_returns,
                    heuristic_target_probs=batch_heuristics,
                    active_masks=batch_masks
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                last_loss = loss.item()
                
            loss_fn.decay_beta()
            print(f"Episode {ep:02d}/{num_episodes} | Steps: {step:02d} | Result: {result_str:<30} | Loss: {last_loss:.4f} | Beta: {loss_fn.beta:.4f}")
        else:
            print(f"Episode {ep:02d}/{num_episodes} | Empty episode")

    # Save final model weights to a separate path to preserve the best checkpoint
    os.makedirs("models", exist_ok=True)
    if best_success_rate == 0.0:
        torch.save(model.state_dict(), "models/multi_agent_model.pth")
        print("--> No best model saved during training. Saved final model to models/multi_agent_model.pth")
    else:
        torch.save(model.state_dict(), "models/multi_agent_model_final.pth")
        print(f"--> Saved final model to models/multi_agent_model_final.pth (Best checkpoint success rate was: {best_success_rate*100:.1f}%)")
        
    print("=============================================================")
    print(f" Training Completed. Overall Success Rate: {success_counts / num_episodes * 100:.1f}%")
    print("=============================================================")

if __name__ == "__main__":
    train_multi_agent()
