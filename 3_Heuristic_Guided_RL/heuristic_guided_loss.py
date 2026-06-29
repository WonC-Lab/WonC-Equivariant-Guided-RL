import torch
import torch.nn as nn
import torch.optim as optim

class HeuristicGuidedLoss(nn.Module):
    """
    Calculates a hybrid loss containing:
    1. Reinforcement Learning Loss (Policy Gradient)
    2. Heuristic Guide Loss (Kullback-Leibler Divergence)
    3. Value Head Loss (Mean Squared Error)

    Mathematical Formulation:
    -------------------------
    The total objective function L(\theta) is defined as:
    
    .. math::
        L(\\theta) = L_{PG}(\\theta) + \\beta \\cdot D_{KL}(P_H(s) \\parallel \\pi_\\theta(s)) + \\frac{1}{2} L_V(\\theta)

    Where:
    - Policy Gradient Loss:
      .. math::
          L_{PG}(\\theta) = - \\frac{1}{B} \\sum_{i=1}^B \\log \\pi_\\theta(a_i | s_i) A_i
      with advantage :math:`A_i = G_i - V_\\theta(s_i)`.
      
    - Heuristic Guidance Regularization (KL Divergence):
      .. math::
          D_{KL}(P_H(s) \\parallel \\pi_\\theta(s)) = \\sum_{a \\in \\mathcal{A}} P_H(a|s) \\log \\left( \\frac{P_H(a|s)}{\\pi_\\theta(a|s)} \\right)
      where :math:`P_H(a|s)` is the heuristic target probability distribution (heuristic_target_probs)
      and :math:`\\pi_\\theta(a|s)` is the policy distribution computed via log_softmax of policy_logits.

    - Value Head Loss (MSE):
      .. math::
          L_V(\\theta) = \\frac{1}{B} \\sum_{i=1}^B (G_i - V_\\theta(s_i))^2

    Theoretical Gradient Derivation:
    --------------------------------
    Since the heuristic policy :math:`P_H(a|s)` does not depend on network weights :math:`\\theta`, taking the 
    gradient of the objective with respect to :math:`\\theta` yields:
    
    .. math::
        \\nabla_\\theta L_{policy}(\\theta) = - \\frac{1}{B} \\sum_{i=1}^B \\sum_{a \\in \\mathcal{A}} \\left[ \\mathbb{I}(a_i = a) A_i + \\beta P_H(a | s_i) \\right] \\nabla_\\theta \\log \\pi_\\theta(a | s_i)
    
    where :math:`\\mathbb{I}` is the indicator function. The term :math:`\\beta P_H(a | s_i)` mathematically functions 
    as a "pseudo-advantage", dynamically regularizing early optimization trajectories to mirror the 
    safe baseline controller.
    
    Beta Coefficient Decay:
    -----------------------
    :math:`\\beta` is geometrically decayed to shift the optimization priority from heuristic supervision 
    to self-play reinforcement learning exploration:
    
    .. math::
        \\beta_{t+1} = \\max(\\beta_t \\cdot \\gamma_{decay}, \\beta_{min})
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

    def forward(self, policy_logits, values, actions, rl_returns, heuristic_target_probs):
        """
        Computes the forward pass of the hybrid heuristic-guided loss function.

        Parameters
        ----------
        policy_logits          : (Batch, NumActions) raw logits from neural net.
                                 Represented as \\pi_\\theta(a|s) after softmax.
        values                 : (Batch, 1) value estimations from neural net.
                                 Represented as V_\\theta(s).
        actions                : (Batch,) integer indexes of chosen actions.
                                 Represented as a_i.
        rl_returns             : (Batch,) scalar rewards/returns.
                                 Represented as G_i.
        heuristic_target_probs : (Batch, NumActions) heuristic recommended probabilities (Must sum to 1.0).
                                 Represented as P_H(a|s).

        Returns
        -------
        total_loss : torch.Tensor
            Combined loss: L_{PG} + \\beta * D_{KL} + 0.5 * L_V
        rl_loss    : float
            Policy gradient loss value (L_{PG})
        kl_loss    : float
            Kullback-Leibler divergence regularization loss value (D_{KL})
        value_loss : float
            Value prediction MSE loss value (L_V)
        """
        # 1. Classical RL Policy Gradient Loss using advantages: -log_prob(a) * Advantage
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]
        
        # Advantage = Return - Value (estimated by Critic)
        with torch.no_grad():
            advantages = rl_returns - values.squeeze(-1)
            
        rl_loss = -torch.mean(selected_log_probs * advantages)

        # 2. Heuristic Guidance Loss via KL Divergence: KL( Heuristic || Agent )
        kl_loss = self.kl_loss_fn(log_probs, heuristic_target_probs)

        # 3. Value Head Loss (MSE)
        import torch.nn.functional as F
        value_loss = F.mse_loss(values.squeeze(-1), rl_returns)

        # 4. Hybrid Total Loss
        total_loss = rl_loss + self.beta * kl_loss + 0.5 * value_loss

        return total_loss, rl_loss.item(), kl_loss.item(), value_loss.item()


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
    
    loss, rl_loss_val, kl_loss_val, val_loss_val = guided_loss_fn(
        logits, values, mock_actions, mock_returns, mock_heuristic_probs
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
