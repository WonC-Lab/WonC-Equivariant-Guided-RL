import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from robotic_env import RoboticArm3DEnv

def train_agent(model, env, num_epochs=150, lr=1e-3, gamma=0.99, sigma=0.1, 
                beta_start=1.0, beta_decay=0.97, beta_min=0.01, restrict_sector=False):
    """
    Trains the policy-value network using baseline-regularized REINFORCE 
    with continuous Gaussian KL regularization against the IK prior.
    Batches updates across 4 episodes per epoch to stabilize updates.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    beta = beta_start
    history = {
        "epoch_rewards": [],
        "epoch_success_rates": [],
        "epoch_losses": [],
        "epoch_kl_losses": []
    }
    
    num_episodes_per_epoch = 4
    
    for epoch in range(num_epochs):
        model.train()
        
        all_loss_pg = []
        all_loss_v = []
        all_loss_kl = []
        
        for _ in range(num_episodes_per_epoch):
            states = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            heuristics = []
            
            obs = env.reset(restrict_sector=restrict_sector)
            done = False
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # Forward pass
                act_mean_t, val_t = model(obs_t)
                act_mean = act_mean_t.squeeze(0)
                val = val_t.squeeze(0)
                
                # Bounding the policy mean to norm <= 1.0 to prevent gradient explosion
                act_mean_norm = torch.norm(act_mean)
                if act_mean_norm > 1.0:
                    act_mean = act_mean * (1.0 / act_mean_norm)
                
                # Sample action
                noise = torch.randn_like(act_mean) * sigma
                action_t = act_mean + noise
                
                # Clip action by norm to preserve SO(3) equivariance
                act_norm = torch.norm(action_t)
                v_max = 1.0
                if act_norm > v_max:
                    action_t = action_t * (v_max / act_norm)
                    
                dist = torch.distributions.Normal(act_mean, sigma)
                log_prob = dist.log_prob(action_t).sum()
                
                # Compute APF (Artificial Potential Field) IK direction
                ee_pos = obs[:, 0]
                target_pos = obs[:, 1]
                obs_pos = obs[:, 2]
                
                # Attractive force towards target
                dir_target = target_pos - ee_pos
                dist_target = np.linalg.norm(dir_target)
                F_att = dir_target / dist_target if dist_target > 1e-6 else np.zeros(3, dtype=np.float32)
                
                # Repulsive force from obstacle
                dir_obs = ee_pos - obs_pos
                dist_obs = np.linalg.norm(dir_obs)
                d0 = 0.35 # Influence distance
                if dist_obs < d0 and dist_obs > 1e-6:
                    eta = 0.15 # Scaling factor
                    F_rep = eta * (1.0 / dist_obs - 1.0 / d0) * (dir_obs / (dist_obs ** 3))
                else:
                    F_rep = np.zeros(3, dtype=np.float32)
                    
                F_net = F_att + F_rep
                F_net_norm = np.linalg.norm(F_net)
                h_mean = F_net / F_net_norm if F_net_norm > 1e-6 else np.zeros(3, dtype=np.float32)
                
                states.append(obs_t)
                actions.append(action_t)
                log_probs.append(log_prob)
                values.append(val)
                heuristics.append(torch.tensor(h_mean, dtype=torch.float32))
                
                obs, reward, done, info = env.step(action_t.detach().numpy())
                rewards.append(reward)
                
            # Process episode
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
                
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.cat(values).view(-1)
            log_probs = torch.stack(log_probs)
            heuristics = torch.stack(heuristics)
            
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
            loss_pg = -(log_probs * advantages).mean()
            loss_v = F.mse_loss(values, returns)
            
            act_means = torch.cat([model(s)[0] for s in states])
            # Bound the policy means in the batch to norm <= 1.0
            act_means_norm = torch.norm(act_means, dim=-1, keepdim=True)
            act_means = torch.where(act_means_norm > 1.0, act_means / act_means_norm, act_means)
            loss_kl = 0.5 * torch.mean(torch.sum((heuristics - act_means)**2, dim=-1))
            
            all_loss_pg.append(loss_pg)
            all_loss_v.append(loss_v)
            all_loss_kl.append(loss_kl)
            
        # Average losses
        mean_loss_pg = torch.stack(all_loss_pg).mean()
        mean_loss_v = torch.stack(all_loss_v).mean()
        mean_loss_kl = torch.stack(all_loss_kl).mean()
        
        # Combined Loss
        total_loss = mean_loss_pg + beta * mean_loss_kl + 0.01 * mean_loss_v
        
        # Optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Decay beta
        beta = max(beta * beta_decay, beta_min)
        
        # Evaluate current policy
        eval_reward, eval_success = evaluate_policy(model, env, restrict_sector=restrict_sector)
        
        history["epoch_rewards"].append(eval_reward)
        history["epoch_success_rates"].append(eval_success)
        history["epoch_losses"].append(total_loss.item())
        history["epoch_kl_losses"].append(mean_loss_kl.item())
        
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Eval Reward: {eval_reward:.2f} | Success: {eval_success:.1f}% | Beta: {beta:.3f} | Total Loss: {total_loss.item():.4f}")
            
    return history

def evaluate_policy(model, env, num_episodes=10, restrict_sector=False):
    """
    Evaluates the model deterministically (without exploration noise) 
    over a number of episodes.
    """
    model.eval()
    total_rewards = []
    successes = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset(restrict_sector=restrict_sector)
            done = False
            ep_reward = 0.0
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                act_mean, _ = model(obs_t)
                action = act_mean.squeeze(0)
                
                # Clip by norm
                act_norm = torch.norm(action)
                if act_norm > 1.0:
                    action = action * (1.0 / act_norm)
                    
                obs, reward, done, info = env.step(action.numpy())
                ep_reward += reward
                
            total_rewards.append(ep_reward)
            successes.append(float(info["success"]))
            
    return np.mean(total_rewards), np.mean(successes) * 100.0
