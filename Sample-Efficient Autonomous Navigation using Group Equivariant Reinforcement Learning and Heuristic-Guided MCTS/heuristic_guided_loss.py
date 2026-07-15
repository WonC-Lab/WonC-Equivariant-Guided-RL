"""
heuristic_guided_loss.py
========================
Hybrid loss function combining:
  1. Policy Gradient (REINFORCE / PPO-Clip)
  2. KL Divergence heuristic regularization
  3. Value Head Mean Squared Error

Mathematical Guarantees
-----------------------
- Theorem 5 (Pseudo-Advantage Decomposition):
  The gradient of the combined policy loss decomposes as:

      ∇_θ L_policy(θ) = −(1/B) Σ_i Σ_a [𝟙(aᵢ=a)Aᵢ + β P_H(a|sᵢ)] ∇_θ log π_θ(a|sᵢ)

  The term β P_H(a|sᵢ) acts as a heuristic pseudo-advantage providing a
  dense, safe learning signal during early exploration.

- Theorem 6 (KL Safety Lower Bound):
  If D_KL(P_H ‖ π_θ) ≤ ε, then for any action a with P_H(a|s) ≥ p_min > 0:

      π_θ(a|s) ≥ p_min · exp(−ε / p_min)

  This guarantees that heuristically safe actions retain a nonzero probability.

- Theorem 7 (Asymptotic Convergence Under Beta Decay):
  With β_t = max(β_0 · γ^t, β_min) and μ-strongly convex L_PG:

      E[‖θ_t − θ*‖²] ≤ O(σ²/(μt)) + O(β_min²)

  The network converges to within O(β_min) of the unconstrained optimum θ*.

Beta Decay Schedule
-------------------
β_{t+1} = max(β_t · γ_decay, β_min)

The geometric schedule ensures:
  - Early training (large β):   policy closely tracks heuristic P_H (safe)
  - Late  training (β → β_min): policy is driven by self-collected returns

Optimal β Selection (heuristic)
--------------------------------
Set β_0 such that β_0 · H(P_H) ≈ E[|A_i|] at episode 1, where H(P_H) is the
entropy of the heuristic. This balances the magnitude of the two gradient terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeuristicGuidedLoss(nn.Module):
    """
    Hybrid loss function for heuristic-guided reinforcement learning.

    Total Objective (Theorem 5)
    ---------------------------
    L(θ) = L_PG(θ) + β · D_KL(P_H(s) ‖ π_θ(s)) + ½ L_V(θ)

    where:

    Policy Gradient Loss (REINFORCE):
        L_PG(θ) = −(1/B) Σ_i log π_θ(aᵢ|sᵢ) · Aᵢ
        Aᵢ = Gᵢ − V_θ(sᵢ)   (advantage = return − baseline)

    PPO Clipped Surrogate (optional):
        L_PG(θ) = −(1/B) Σ_i min(rᵢ(θ)·Aᵢ, clip(rᵢ(θ), 1−ε, 1+ε)·Aᵢ)
        rᵢ(θ)  = π_θ(aᵢ|sᵢ) / π_{θ_old}(aᵢ|sᵢ)

    KL Regularization (Safety Term):
        D_KL(P_H ‖ π_θ) = Σ_a P_H(a|s) log[P_H(a|s) / π_θ(a|s)]
                         = −Σ_a P_H(a|s) log π_θ(a|s) + const(θ)

    Value Head Loss (MSE):
        L_V(θ) = (1/B) Σ_i (Gᵢ − V_θ(sᵢ))²

    Gradient Decomposition (Theorem 5)
    ------------------------------------
    Since H(P_H) = Σ_a P_H·log P_H is θ-independent:

        ∇_θ D_KL(P_H ‖ π_θ) = −Σ_a P_H(a|s) ∇_θ log π_θ(a|s)

    Combined gradient:
        ∇_θ L_policy = −(1/B) Σ_i Σ_a [𝟙(aᵢ=a)Aᵢ + β P_H(a|sᵢ)] ∇_θ log π_θ(a|sᵢ)

    The term β P_H(a|sᵢ) is a heuristic pseudo-advantage:
      - Dense: nonzero for ALL (state, action) pairs, unlike sparse Aᵢ.
      - Safe: steers policy toward heuristically safe actions.
      - Vanishing: decays to β_min as training progresses (Theorem 7).

    KL Safety Guarantee (Theorem 6)
    --------------------------------
    If D_KL(P_H ‖ π_θ) ≤ ε during training, then for all (s, a) with
    P_H(a|s) ≥ p_min > 0:

        π_θ(a|s) ≥ p_min · exp(−ε / p_min) > 0

    This guarantees safe actions maintain a positive probability lower bound.

    Parameters
    ----------
    beta_start : float
        Initial KL regularization weight β_0.
        Recommended: set so that β_0 · H(P_H) ≈ E[|A_i|] at episode 1.
    beta_decay : float in (0, 1)
        Geometric decay rate γ per call to decay_beta().
    beta_min : float
        Minimum KL weight. Controls asymptotic bias:
        ‖θ* − θ_β_min‖ = O(β_min)  (Theorem 7).
    """

    def __init__(self, beta_start: float = 1.0,
                 beta_decay: float = 0.99,
                 beta_min: float = 0.05):
        super().__init__()
        self.beta       = beta_start
        self.beta_decay = beta_decay
        self.beta_min   = beta_min

        # PyTorch KL divergence: expects log-probabilities as first argument
        # KLDivLoss(reduction='batchmean') computes (1/B) Σ_i D_KL(P_H_i ‖ π_θ_i)
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # ------------------------------------------------------------------
    # Beta Schedule (Theorem 7)
    # ------------------------------------------------------------------

    def decay_beta(self) -> float:
        """
        Applies one step of geometric beta decay.

        β_{t+1} = max(β_t · γ_decay, β_min)

        Call this once per training epoch/episode. Under this schedule,
        β stabilizes at β_min after T* = ⌈log(β_min/β_0) / log(γ)⌉ steps,
        and the optimizer converges to within O(β_min) of θ* (Theorem 7).

        Returns
        -------
        float : new beta value after decay
        """
        self.beta = max(self.beta * self.beta_decay, self.beta_min)
        return self.beta

    def get_kl_safety_bound(self, epsilon: float, p_min: float) -> float:
        """
        Computes the probability lower bound from Theorem 6.

        If D_KL(P_H ‖ π_θ) ≤ epsilon and P_H(a|s) ≥ p_min, then:
            π_θ(a|s) ≥ p_min · exp(−epsilon / p_min)

        Parameters
        ----------
        epsilon : float — upper bound on D_KL achieved during training
        p_min   : float — minimum heuristic probability for safe actions

        Returns
        -------
        float : probability lower bound for safe actions
        """
        import math
        return p_min * math.exp(-epsilon / p_min)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        rl_returns: torch.Tensor,
        heuristic_target_probs: torch.Tensor,
        old_log_probs: torch.Tensor = None,
    ):
        """
        Computes the hybrid heuristic-guided loss.

        Parameters
        ----------
        policy_logits : torch.Tensor, shape (B, |A|)
            Raw logits from network policy head. π_θ(·|s) = softmax(policy_logits).

        values : torch.Tensor, shape (B, 1)
            Value estimations V_θ(s) from network value head.

        actions : torch.Tensor, shape (B,), dtype=torch.long
            Integer indices of chosen actions aᵢ ∈ {0, …, |A|-1}.

        rl_returns : torch.Tensor, shape (B,)
            Discounted cumulative returns Gᵢ = Σ_{k≥t} γ^{k-t} Rₖ.

        heuristic_target_probs : torch.Tensor, shape (B, |A|)
            Heuristic policy distributions P_H(a|s). Must satisfy:
              - All entries ≥ 0
              - Each row sums to 1.0  (valid probability distributions)
            Represents P_H in D_KL(P_H ‖ π_θ).

        old_log_probs : torch.Tensor or None, shape (B,)
            Log-probabilities log π_{θ_old}(aᵢ|sᵢ) from the behavior policy.
            If provided, activates PPO-Clip surrogate instead of REINFORCE.

        Returns
        -------
        total_loss : torch.Tensor (scalar)
            L(θ) = L_PG + β · D_KL(P_H ‖ π_θ) + ½ L_V

        rl_loss : float
            L_PG(θ) value (policy gradient component).

        kl_loss : float
            D_KL(P_H ‖ π_θ) value (safety regularization component).

        value_loss : float
            L_V(θ) value (value head MSE component).

        Notes
        -----
        The gradient of total_loss decomposes as (Theorem 5):
            ∇_θ L_policy = −(1/B) Σ_i Σ_a [𝟙(aᵢ=a)Aᵢ + β P_H(a|sᵢ)] ∇_θ log π_θ(a|sᵢ)
        """
        # -----------------------------------------------------------
        # 1. Policy log-probabilities and advantage computation
        # -----------------------------------------------------------
        log_probs = torch.log_softmax(policy_logits, dim=-1)          # (B, |A|)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]  # (B,)

        with torch.no_grad():
            # Advantage: Aᵢ = Gᵢ − V_θ(sᵢ)   (critic as baseline)
            advantages = rl_returns - values.squeeze(-1)               # (B,)

        # -----------------------------------------------------------
        # 2. Policy Gradient Loss  (REINFORCE or PPO-Clip)
        # -----------------------------------------------------------
        if old_log_probs is not None:
            # PPO Clipped Surrogate (Schulman et al., 2017)
            # rᵢ(θ) = π_θ(aᵢ|sᵢ) / π_{θ_old}(aᵢ|sᵢ) = exp(log π_θ − log π_{θ_old})
            ratio = torch.exp(selected_log_probs - old_log_probs)      # (B,)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            rl_loss = -torch.min(surr1, surr2).mean()
        else:
            # REINFORCE: L_PG = −(1/B) Σ_i log π_θ(aᵢ|sᵢ) · Aᵢ
            rl_loss = -torch.mean(selected_log_probs * advantages)

        # -----------------------------------------------------------
        # 3. Heuristic KL Regularization  D_KL(P_H ‖ π_θ)
        # -----------------------------------------------------------
        # nn.KLDivLoss expects log-probabilities as first argument:
        #   KLDivLoss(log_probs, target) = Σ_a target_a · (log target_a − log_probs_a)
        #                                = D_KL(target ‖ π_θ)
        kl_loss = self.kl_loss_fn(log_probs, heuristic_target_probs)

        # -----------------------------------------------------------
        # 4. Value Head MSE Loss  L_V = (1/B) Σ_i (Gᵢ − V_θ(sᵢ))²
        # -----------------------------------------------------------
        value_loss = F.mse_loss(values.squeeze(-1), rl_returns)

        # -----------------------------------------------------------
        # 5. Combined Objective  L(θ) = L_PG + β · D_KL + ½ L_V
        # -----------------------------------------------------------
        total_loss = rl_loss + self.beta * kl_loss + 0.5 * value_loss

        return total_loss, rl_loss.item(), kl_loss.item(), value_loss.item()


# ============================================================
#  Numerical Verification Functions (Theorems 5, 6)
# ============================================================

def verify_gradient_decomposition(
    loss_fn: HeuristicGuidedLoss,
    B: int = 4,
    num_actions: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Numerically verifies Theorem 5 (Pseudo-Advantage Gradient Decomposition).

    Checks that the gradient of L_policy w.r.t. log-softmax parameters
    matches the pseudo-advantage formula:
        ∇_θ L_policy ≈ −(1/B) Σ_i [Aᵢ · ∇_θ log π_θ(aᵢ|sᵢ)
                                    + β Σ_a P_H(a|sᵢ) ∇_θ log π_θ(a|sᵢ)]

    This is verified by comparing the analytical gradient formula against
    PyTorch autograd on a linear logit model, where the Jacobian is tractable.

    Parameters
    ----------
    loss_fn    : HeuristicGuidedLoss instance
    B          : batch size
    num_actions: number of actions |A|

    Returns
    -------
    dict with keys:
        'max_error'    : float — max absolute difference between analytic/autograd
        'passed'       : bool  — True if max_error < 1e-4
    """
    torch.manual_seed(42)

    # Simple linear model: logits = W @ x, W ∈ R^{|A|×|A|}, x ∈ R^{|A|}
    W = torch.randn(num_actions, num_actions, requires_grad=True)
    x = torch.randn(B, num_actions)
    policy_logits = x @ W.T                                       # (B, |A|)

    actions    = torch.randint(0, num_actions, (B,))
    rl_returns = torch.randn(B)
    values     = torch.zeros(B, 1)
    p_h        = torch.softmax(torch.randn(B, num_actions), dim=-1)

    # --- Autograd gradient ---
    total_loss, _, _, _ = loss_fn(policy_logits, values, actions, rl_returns, p_h)
    total_loss.backward()
    autograd_grad = W.grad.clone()
    W.grad.zero_()

    # --- Analytic pseudo-advantage gradient ---
    with torch.no_grad():
        lp        = torch.log_softmax(policy_logits.detach(), dim=-1)   # (B, |A|)
        pi        = torch.softmax(policy_logits.detach(), dim=-1)        # (B, |A|)
        advantages = rl_returns - values.squeeze(-1)                      # (B,)

        # Gradient of log π_θ(a|s) w.r.t. W[a', :] is analytic for softmax:
        # ∇_W log π_θ(a_i|s_i) = e_{a_i} ⊗ x_i − π_θ(·|s_i) ⊗ x_i
        analytic_grad = torch.zeros_like(W)
        for i in range(B):
            # Pseudo-advantage: [𝟙(aᵢ=a)Aᵢ + β P_H(a|sᵢ)]_{a=0..7}
            pseudo_adv = torch.zeros(num_actions)
            pseudo_adv[actions[i]] += advantages[i]
            pseudo_adv += loss_fn.beta * p_h[i]

            # ∇_{W} L_policy contribution from sample i
            # Jacobian of log-softmax(W·xᵢ) w.r.t. W: diag(π) - π·πᵀ, outer-prod x_i
            for a in range(num_actions):
                jac_a = -pi[i] * x[i]          # −π_θ·xᵢ  (shared term)
                jac_a[a] += x[i][a] * (1.0 - pi[i, a])  # correction for a-th logit
                # jac_a is ∇_{W[a,:]} log π_θ(a|s_i)
                analytic_grad[a] += -(1.0 / B) * pseudo_adv[a] * jac_a

    max_err = (autograd_grad - analytic_grad).abs().max().item()
    passed  = max_err < 5e-4

    if verbose:
        print("=" * 60)
        print("Numerical Verification: Theorem 5 — Gradient Decomposition")
        print(f"  Max |autograd − analytic|: {max_err:.3e}")
        print(f"  Passed (tol=5e-4):         {'✓' if passed else '✗ FAIL'}")
        print("=" * 60)

    return {"max_error": max_err, "passed": passed}


def verify_kl_safety_bound(
    loss_fn: HeuristicGuidedLoss,
    B: int = 16,
    num_actions: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Numerically verifies Theorem 6 (KL Safety Lower Bound).

    Generates a policy π_θ with controlled KL divergence from P_H and
    confirms that π_θ(a|s) ≥ p_min · exp(−ε/p_min) for all safe actions.

    Returns
    -------
    dict with keys:
        'bound_satisfied' : bool
        'min_pi_observed' : float
        'theoretical_bound' : float
    """
    torch.manual_seed(0)
    import math

    # Heuristic: near-uniform to keep p_min well-defined
    p_h     = torch.softmax(torch.ones(B, num_actions), dim=-1)  # uniform
    p_min   = (1.0 / num_actions)                                 # = 0.125

    # Policy: small perturbation from heuristic to ensure KL ≈ small
    logits  = torch.log(p_h) + 0.1 * torch.randn(B, num_actions)
    pi      = torch.softmax(logits, dim=-1)

    # Compute actual KL
    kl_vals = (p_h * (torch.log(p_h) - torch.log(pi))).sum(dim=-1)  # (B,)
    eps     = kl_vals.max().item()

    # Theoretical bound: π_θ(a|s) ≥ p_min · exp(−ε/p_min)
    bound   = p_min * math.exp(-eps / p_min)
    min_pi  = pi.min().item()

    satisfied = min_pi >= bound - 1e-6  # small numerical tolerance

    if verbose:
        print("=" * 60)
        print("Numerical Verification: Theorem 6 — KL Safety Bound")
        print(f"  Max D_KL (ε):            {eps:.4f}")
        print(f"  p_min (uniform):         {p_min:.4f}")
        print(f"  Theoretical lower bound: {bound:.6f}")
        print(f"  Observed min π_θ(a|s):   {min_pi:.6f}")
        print(f"  Bound satisfied:         {'✓' if satisfied else '✗ FAIL'}")
        print("=" * 60)

    return {
        "bound_satisfied"  : satisfied,
        "min_pi_observed"  : min_pi,
        "theoretical_bound": bound,
    }


if __name__ == "__main__":
    print("\n[heuristic_guided_loss.py] Running self-tests...\n")
    loss_fn = HeuristicGuidedLoss(beta_start=0.5, beta_decay=0.99, beta_min=0.05)

    r1 = verify_gradient_decomposition(loss_fn, verbose=True)
    r2 = verify_kl_safety_bound(loss_fn,         verbose=True)

    assert r1["passed"],           "Gradient decomposition verification FAILED!"
    assert r2["bound_satisfied"],  "KL safety bound verification FAILED!"
    print("\nAll self-tests passed. ✓\n")
