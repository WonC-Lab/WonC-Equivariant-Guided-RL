import torch
import torch.nn as nn
import torch.optim as optim

class HeuristicGuidedLoss(nn.Module):
    """
    Calculates a hybrid loss containing:
    1. Reinforcement Learning Loss (Policy Gradient)
    2. Heuristic Guide Loss (Kullback-Leibler Divergence)
    """
    def __init__(self, beta_start=1.0, beta_decay=0.99, beta_min=0.01):
        super().__init__()
        self.beta = beta_start
        self.beta_decay = beta_decay
        self.beta_min = beta_min
        
        # PyTorch KL Divergence Loss
        # log_target=False ensures target input is expected as standard probabilities
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def decay_beta(self):
        """
        Decays the heuristic regularization coefficient after each training epoch/step.
        """
        self.beta = max(self.beta * self.beta_decay, self.beta_min)

    def forward(self, policy_logits, actions, rl_returns, heuristic_target_probs):
        """
        Parameters
        ----------
        policy_logits          : (Batch, NumActions) raw logits from neural net
        actions                : (Batch,) integer indexes of chosen actions
        rl_returns             : (Batch,) scalar rewards/returns (G_t)
        heuristic_target_probs : (Batch, NumActions) heuristic recommended probabilities (Must sum to 1.0)
        """
        # 1. Classical RL Policy Gradient Loss: -log_prob(a) * Return
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]
        
        # Negative sign since we perform gradient descent to maximize returns
        rl_loss = -torch.mean(selected_log_probs * rl_returns)

        # 2. Heuristic Guidance Loss via KL Divergence: KL( Heuristic || Agent )
        # nn.KLDivLoss takes log-probabilities as the input and standard probabilities as the target
        kl_loss = self.kl_loss_fn(log_probs, heuristic_target_probs)

        # 3. Hybrid Total Loss
        total_loss = rl_loss + self.beta * kl_loss

        return total_loss, rl_loss.item(), kl_loss.item()


# Simple verification
if __name__ == "__main__":
    from equivariant_models import D4EquivariantNet

    print("Verifying Heuristic Guided Training Step...")

    # 1. Initialize Network & Hybrid Loss
    batch_size = 4
    board_size = 13
    num_actions = board_size * board_size
    
    model = D4EquivariantNet(board_size=board_size, in_channels=3, num_filters=16, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    guided_loss_fn = HeuristicGuidedLoss(beta_start=1.0, beta_decay=0.95, beta_min=0.05)

    # 2. Mock Training Batch Tensors
    mock_inputs = torch.randn(batch_size, 3, board_size, board_size)
    mock_actions = torch.randint(0, num_actions, (batch_size,))
    mock_returns = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)

    # Simulate heuristic recommendation (e.g. uniform with slight noise for testing)
    raw_heuristics = torch.rand(batch_size, num_actions)
    mock_heuristic_probs = torch.softmax(raw_heuristics, dim=-1) # must sum to 1.0 per row

    # 3. Perform a single training step
    model.train()
    optimizer.zero_grad()
    
    logits, values = model(mock_inputs)
    
    loss, rl_loss_val, kl_loss_val = guided_loss_fn(
        logits, mock_actions, mock_returns, mock_heuristic_probs
    )
    
    loss.backward()
    optimizer.step()
    
    # 4. Decay beta coefficient
    initial_beta = guided_loss_fn.beta
    guided_loss_fn.decay_beta()
    decayed_beta = guided_loss_fn.beta

    print("Training step completed successfully!")
    print(f"  Total Hybrid Loss     : {loss.item():.4f}")
    print(f"  RL Policy Gradient Loss: {rl_loss_val:.4f}")
    print(f"  KL Heuristic Guide Loss: {kl_loss_val:.4f}")
    print(f"  Beta Coefficient      : {initial_beta:.4f} -> {decayed_beta:.4f}")
    
    assert loss.item() > 0, "Loss calculation anomaly detected!"
    assert decayed_beta < initial_beta, "Beta decay failed!"
    print("OK: Heuristic guided loss verified successfully.")
