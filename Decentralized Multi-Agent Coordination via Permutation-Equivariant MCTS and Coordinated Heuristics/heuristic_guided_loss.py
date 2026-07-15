import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAgentHeuristicGuidedLoss(nn.Module):
    """
    Decentralized Multi-Agent Heuristic-Guided PPO Loss Function.
    Applies PPO clipping logic and KL regularization against coordinated heuristics,
    masking out inactive/crashed/completed agents.
    
    Mathematical formulation:
    L_total = L_CLIP + \beta * D_KL(P_H || \pi_\theta) + 0.5 * L_V
    """
    def __init__(self, beta_start=1.0, beta_decay=0.99, beta_min=0.01, clip_eps=0.2):
        super().__init__()
        self.beta = beta_start
        self.beta_decay = beta_decay
        self.beta_min = beta_min
        self.clip_eps = clip_eps
        
        # batchmean computes the mathematical KL divergence properly
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def decay_beta(self):
        self.beta = max(self.beta * self.beta_decay, self.beta_min)

    def forward(self, policy_logits, values, actions, old_log_probs, rl_returns, heuristic_target_probs, active_masks):
        """
        Parameters:
          policy_logits: (Batch, M, 8)
          values: (Batch, M, 1)
          actions: (Batch, M)
          old_log_probs: (Batch, M)
          rl_returns: (Batch, M)
          heuristic_target_probs: (Batch, M, 8)
          active_masks: (Batch, M) - binary tensor (1.0 for active, 0.0 for inactive)
        """
        B, M, _ = policy_logits.shape
        
        # Flatten inputs to simplify masking
        logits_flat = policy_logits.view(B * M, 8)
        values_flat = values.view(B * M)
        actions_flat = actions.view(B * M)
        old_log_probs_flat = old_log_probs.view(B * M)
        returns_flat = rl_returns.view(B * M)
        heuristic_flat = heuristic_target_probs.view(B * M, 8)
        masks_flat = active_masks.view(B * M)
        
        # Filter active agent experiences
        active_indices = torch.where(masks_flat > 0.5)[0]
        if len(active_indices) == 0:
            # Fallback if no active experiences in batch
            return torch.tensor(0.0, requires_grad=True), 0.0, 0.0, 0.0
            
        logits_act = logits_flat[active_indices]
        values_act = values_flat[active_indices]
        actions_act = actions_flat[active_indices]
        old_log_probs_act = old_log_probs_flat[active_indices]
        returns_act = returns_flat[active_indices]
        heuristic_act = heuristic_flat[active_indices]
        
        # Compute policy representations
        log_probs = torch.log_softmax(logits_act, dim=-1)
        probs = torch.softmax(logits_act, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(actions_act)), actions_act]
        
        # 1. PPO Clipped Policy Gradient Loss
        advantages = returns_act - values_act.detach()
        ratios = torch.exp(selected_log_probs - old_log_probs_act)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        # 2. KL Divergence Heuristic Guidance Loss
        # PyTorch KLDiv expects input in log space and target in linear probability space
        kl_loss = self.kl_loss_fn(log_probs, heuristic_act)
        
        # 3. Value Loss (MSE)
        value_loss = F.mse_loss(values_act, returns_act)
        
        # 4. Total Combined Loss
        total_loss = policy_loss + self.beta * kl_loss + 0.5 * value_loss
        
        return total_loss, policy_loss.item(), kl_loss.item(), value_loss.item()
