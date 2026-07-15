"""
equivariant_models.py
=====================
PyTorch implementation of D4-Equivariant neural networks for grid-based
reinforcement learning, with numerical verification of all theoretical guarantees.

Mathematical Guarantees
-----------------------
- Theorem 1 (Group Frame Averaging): The frame-averaged operator F[h] is
  G-equivariant for *any* base function h, proved by the rearrangement lemma.
- Theorem 2 (Softmax Equivariance): Policy π_θ satisfies
  π_θ(g·a | g·s) = π_θ(a | s) ∀ g ∈ D4.
- Corollary 1 (Value Invariance): V_θ(g·s) = V_θ(s) ∀ g ∈ D4.

Reference
---------
Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks.
ICML 2016. https://arxiv.org/abs/1602.07576
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class D4GroupAction:
    """
    Implements the action of the Dihedral Group D_4 on 2D grid tensors.

    Group Structure
    ---------------
    D_4 = <r, m | r^4 = m^2 = e, mrm = r^{-1}>

    The 8 elements are indexed as follows:
        0: identity   e
        1: rotation   r      (90° CCW)
        2: rotation   r^2   (180°)
        3: rotation   r^3   (270° CCW = 90° CW)
        4: reflection m      (horizontal flip)
        5: reflection mr     (flip + 90° CCW)
        6: reflection mr^2   (flip + 180°)
        7: reflection mr^3   (flip + 270° CCW)

    Group Law
    ---------
    Each element g ∈ D_4 is encoded as (flip ∈ {0,1}, rot_k ∈ {0,1,2,3}).
    Composition: (f1, k1) · (f2, k2):
        if f1 == 0: (f2, (k1+k2) % 4)
        if f1 == 1: (1-f2 if... ) — handled implicitly by sequential ops.

    Input/Output Shape
    ------------------
    All methods operate on tensors of shape (B, C, H, W).
    """

    @staticmethod
    def apply_action(x: torch.Tensor, action_idx: int) -> torch.Tensor:
        """
        Applies group element g_{action_idx} to a grid tensor.

        The representation ρ_S: D_4 → GL(R^{C×H×W}) is:
            ρ_S(g)(x) = rot90(flip(x), k)   for g = (flip, k)

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)
        action_idx : int in [0, 7]

        Returns
        -------
        torch.Tensor of same shape as x
        """
        rot_k = action_idx % 4
        flip  = action_idx // 4
        out = x
        if flip == 1:
            out = torch.flip(out, dims=[-1])   # reflect about vertical axis
        if rot_k > 0:
            out = torch.rot90(out, k=rot_k, dims=[-2, -1])
        return out

    @staticmethod
    def apply_inverse_action(x: torch.Tensor, action_idx: int) -> torch.Tensor:
        """
        Applies the group inverse g_{action_idx}^{-1} to a grid tensor.

        The inverse of (flip, k) is (flip, (4-k) % 4) when flip=0,
        and (flip, k) is self-inverse when flip=1 (reflections are involutions).
        Operationally: undo rotation first, then undo reflection.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)
        action_idx : int in [0, 7]

        Returns
        -------
        torch.Tensor of same shape as x
        """
        rot_k = action_idx % 4
        flip  = action_idx // 4
        out = x
        if rot_k > 0:
            out = torch.rot90(out, k=4 - rot_k, dims=[-2, -1])
        if flip == 1:
            out = torch.flip(out, dims=[-1])
        return out

    @staticmethod
    def get_action_permutation(action_idx: int) -> list:
        """
        Computes the permutation σ_g of the 8 action directions induced by
        group element g = D4[action_idx].

        This defines the representation ρ_A: D_4 → S_8 on the action space.
        For a rotation/reflection g, action vector v is mapped to g(v).

        Action Encoding (8-directional movement):
            0: (-1,  0)  Up          4: (-1, -1)  Up-Left
            1: ( 1,  0)  Down        5: (-1,  1)  Up-Right
            2: ( 0, -1)  Left        6: ( 1, -1)  Down-Left
            3: ( 0,  1)  Right       7: ( 1,  1)  Down-Right

        Returns
        -------
        perm : list of int, length 8
            perm[a] = index of action g·a in the canonical action table.
        """
        action_vectors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        rot_k = action_idx % 4
        flip  = action_idx // 4

        perm = []
        for dr, dc in action_vectors:
            if flip == 1:
                dc = -dc
            # rot90 CCW: (dr, dc) → (-dc, dr)
            for _ in range(rot_k):
                dr, dc = -dc, dr

            found = False
            for idx, vec in enumerate(action_vectors):
                if vec == (dr, dc):
                    perm.append(idx)
                    found = True
                    break
            if not found:
                raise ValueError(f"Transformed vector ({dr}, {dc}) not in action space.")
        return perm


class EquivariantConv2d(nn.Module):
    """
    D_4-Equivariant Convolutional Layer via Group Frame Averaging.

    Theory (Theorem 1)
    ------------------
    For a standard (non-equivariant) convolution h_φ, the frame-averaged operator:

        F[h_φ](x) = (1/|G|) Σ_{g ∈ G} ρ(g)^{-1} · h_φ(ρ(g) · x)

    is guaranteed to be G-equivariant regardless of φ. This is proved by the
    orbit rearrangement lemma: left-multiplication by g' is a bijection on G.

    Computational Complexity
    ------------------------
    Cost: 8× forward passes through the base convolution, then averaged.
    Memory: O(8 × B × C × H × W) for intermediate feature maps.
    
    Weight Sharing
    --------------
    All 8 group-transformed views share the **same** parameters φ.
    This is the key mechanism: equivariance is enforced architecturally,
    not by data augmentation, saving a factor of |G| = 8 in training data.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        F[h_φ](x) = (1/8) Σ_{g ∈ D4} ρ(g)^{-1} · h_φ(ρ(g) · x)

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)

        Returns
        -------
        equivariant_features : torch.Tensor, shape (B, out_channels, H, W)
            Guaranteed to satisfy:  out(g·x) = g · out(x)  ∀ g ∈ D4.
        """
        outputs = []
        for i in range(8):
            # Step 1: Transform input by g_i  →  ρ(g_i) · x
            transformed = D4GroupAction.apply_action(x, i)
            # Step 2: Apply shared conv  →  h_φ(ρ(g_i) · x)
            features = self.conv(transformed)
            # Step 3: Apply inverse transform  →  ρ(g_i)^{-1} · h_φ(ρ(g_i) · x)
            realigned = D4GroupAction.apply_inverse_action(features, i)
            outputs.append(realigned)

        # Step 4: Average over group  →  (1/|G|) Σ_g  (group projection)
        return torch.mean(torch.stack(outputs, dim=0), dim=0)


class D4EquivariantNet(nn.Module):
    """
    D4-Equivariant Actor-Critic Policy-Value Network.

    Architecture
    ------------
    Uses Group Frame Averaging (Theorem 1) to enforce:
      - Policy Equivariance (Theorem 2):  π_θ(g·a | g·s) = π_θ(a | s)
      - Value Invariance   (Corollary 1): V_θ(g·s)        = V_θ(s)

    The implementation uses the "orbit batching" trick for efficiency:
    all 8 group-transformed views are stacked into a single batch dimension
    and processed in parallel, then averaged.

    Forward Pass Steps
    ------------------
    1. Stack 8 group-transformed copies: x_batched ∈ R^{8B × C × H × W}
    2. Forward through shared StandardCNN backbone
    3. Reshape to (8, B, num_actions) and (8, B, 1)
    4. Apply action permutation σ_g to realign policy logits
    5. Average over group dimension → equivariant logits, invariant value

    Inputs
    ------
    x : torch.Tensor, shape (B, in_channels, H, W)

    Outputs
    -------
    policy_logits : torch.Tensor, shape (B, 8)
        D4-equivariant raw logits. Apply softmax for policy probabilities.
    value : torch.Tensor, shape (B, 1)
        D4-invariant state value estimate in [-1, 1].
    """

    def __init__(self, board_size: int = 13, in_channels: int = 3,
                 num_filters: int = 64, num_layers: int = 3):
        super().__init__()
        self.board_size = board_size
        self.base_net = StandardCNN(board_size, in_channels, num_filters, num_layers)
        # Precompute ρ_A(g): action permutations for all 8 group elements
        self.perms = [D4GroupAction.get_action_permutation(i) for i in range(8)]

    def forward(self, x: torch.Tensor):
        """
        Equivariant forward pass via orbit batching.

        Invariant: output policy satisfies π_θ(g·a|g·s) = π_θ(a|s) ∀g ∈ D4.
        Invariant: output value satisfies V_θ(g·s) = V_θ(s) ∀g ∈ D4.
        """
        B = x.shape[0]

        # 1. Orbit batching: stack all 8 symmetry views  →  shape (8B, C, H, W)
        x_batched = torch.cat(
            [D4GroupAction.apply_action(x, i) for i in range(8)], dim=0
        )

        # 2. Single parallel forward pass through shared backbone
        logits_batched, values_batched = self.base_net(x_batched)

        # 3. Reshape to separate group dimension
        logits_g = logits_batched.view(8, B, 8)   # (8, B, |A|)
        values_g  = values_batched.view(8, B, 1)  # (8, B,  1 )

        # 4. Realign policy logits: apply inverse action permutation σ_{g_i}^{-1}
        #    so that logits_i[b, a] → logits in the *original* action frame
        realigned = []
        for i in range(8):
            perm_i = self.perms[i]           # σ_{g_i}: A → A
            realigned.append(logits_g[i][:, perm_i])  # (B, 8)

        # 5. Group average (orbit projection)  →  equivariant/invariant outputs
        policy_logits = torch.mean(torch.stack(realigned, dim=0), dim=0)  # (B, 8)
        value         = torch.mean(values_g, dim=0)                        # (B, 1)

        return policy_logits, value


class StandardCNN(nn.Module):
    """
    Standard (non-equivariant) CNN backbone.

    Used as the shared sub-network inside D4EquivariantNet and as the
    ablation baseline in experiments.

    Architecture
    ------------
    - Backbone: num_layers stacked (Conv2d → BatchNorm2d → ReLU) blocks
    - Policy head: AdaptiveAvgPool → Flatten → Linear(num_filters, 8)
    - Value head:  AdaptiveAvgPool → Flatten → Linear → ReLU → Linear → Tanh

    No equivariance guarantees are provided for this class alone.
    Equivariance emerges only when wrapped by D4EquivariantNet.
    """

    def __init__(self, board_size: int = 13, in_channels: int = 3,
                 num_filters: int = 64, num_layers: int = 3):
        super().__init__()
        self.board_size = board_size

        layers = [
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*layers)

        # Policy Head: equivariant channel outputs → 8 action logits
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 8),
        )

        # Value Head: aggregated features → scalar ∈ [-1, 1]
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        features      = self.backbone(x)
        policy_logits = self.policy_head(features)
        value         = self.value_head(features)
        return policy_logits, value


# ============================================================
#  Numerical Verification Functions (Theorems 1, 2, Corollary 1)
# ============================================================

@torch.no_grad()
def verify_policy_equivariance(
    model: D4EquivariantNet,
    board_size: int = 13,
    in_channels: int = 3,
    atol: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Numerically verifies Theorem 2: π_θ(g·a | g·s) = π_θ(a | s) ∀ g ∈ D4.

    For each of the 8 group elements g:
      1. Sample a random state s.
      2. Compute π_θ(·|s)  and  π_θ(·|g·s).
      3. Permute π_θ(·|s) by σ_g (the action permutation for g).
      4. Assert that the permuted distribution matches π_θ(·|g·s) within atol.

    Parameters
    ----------
    model       : D4EquivariantNet (eval mode recommended)
    board_size  : spatial grid size H = W
    in_channels : number of input channels
    atol        : absolute tolerance for numerical comparison
    verbose     : if True, prints per-group max absolute error

    Returns
    -------
    results : dict with keys:
        'all_passed'     : bool — True if all 8 group elements pass
        'max_errors'     : list of float — max |error| per group element
        'mean_max_error' : float — average of max errors across D4
    """
    model.eval()
    device = next(model.parameters()).device

    # Random test state
    s = torch.randn(1, in_channels, board_size, board_size, device=device)

    # Compute policy on original state
    logits_s, _ = model(s)
    pi_s = torch.softmax(logits_s, dim=-1).squeeze(0)  # shape (8,)

    perms = [D4GroupAction.get_action_permutation(i) for i in range(8)]

    max_errors = []
    all_passed  = True

    if verbose:
        print("=" * 60)
        print("Numerical Verification: Theorem 2 - Policy Equivariance")
        print(f"{'Group element':>20} | {'Max |error|':>14} | {'Pass?':>6}")
        print("-" * 60)

    group_names = ["e", "r", "r²", "r³", "m", "mr", "mr²", "mr³"]

    for i, (perm, name) in enumerate(zip(perms, group_names)):
        g_s = D4GroupAction.apply_action(s, i)
        logits_gs, _ = model(g_s)
        pi_gs = torch.softmax(logits_gs, dim=-1).squeeze(0)  # π_θ(·|g·s)

        # Expected: π_θ(g·a|g·s) should equal π_θ(a|s)
        # i.e., pi_gs[perm[a]] == pi_s[a]  ∀ a  ↔  pi_gs == pi_s[inv_perm]
        pi_s_permuted = pi_s[perm]             # (π_θ(g·a|s))_{a=0..7}
        error = (pi_gs - pi_s_permuted).abs()
        max_err = error.max().item()
        passed  = max_err < atol
        if not passed:
            all_passed = False
        max_errors.append(max_err)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  g = {name:>6}           | {max_err:>14.2e} | {status:>6}")

    mean_err = float(sum(max_errors) / len(max_errors))
    if verbose:
        print("-" * 60)
        print(f"  Mean max error: {mean_err:.2e}   |  All passed: {all_passed}")
        print("=" * 60)

    return {
        "all_passed"     : all_passed,
        "max_errors"     : max_errors,
        "mean_max_error" : mean_err,
    }


@torch.no_grad()
def verify_value_invariance(
    model: D4EquivariantNet,
    board_size: int = 13,
    in_channels: int = 3,
    atol: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Numerically verifies Corollary 1: V_θ(g·s) = V_θ(s) ∀ g ∈ D4.

    Parameters
    ----------
    model       : D4EquivariantNet (eval mode recommended)
    board_size  : spatial grid size H = W
    in_channels : number of input channels
    atol        : absolute tolerance for numerical comparison
    verbose     : if True, prints per-group max absolute error

    Returns
    -------
    results : dict with keys:
        'all_passed'  : bool
        'max_errors'  : list of float
        'mean_max_error' : float
    """
    model.eval()
    device = next(model.parameters()).device

    s = torch.randn(1, in_channels, board_size, board_size, device=device)
    _, v_s = model(s)
    v_s = v_s.item()

    max_errors = []
    all_passed  = True
    group_names = ["e", "r", "r²", "r³", "m", "mr", "mr²", "mr³"]

    if verbose:
        print("=" * 60)
        print("Numerical Verification: Corollary 1 - Value Invariance")
        print(f"{'Group element':>20} | {'|V(g·s)-V(s)|':>14} | {'Pass?':>6}")
        print("-" * 60)

    for i, name in enumerate(group_names):
        g_s = D4GroupAction.apply_action(s, i)
        _, v_gs = model(g_s)
        err     = abs(v_gs.item() - v_s)
        passed  = err < atol
        if not passed:
            all_passed = False
        max_errors.append(err)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  g = {name:>6}           | {err:>14.2e} | {status:>6}")

    mean_err = float(sum(max_errors) / len(max_errors))
    if verbose:
        print("-" * 60)
        print(f"  Mean error: {mean_err:.2e}   |  All passed: {all_passed}")
        print("=" * 60)

    return {
        "all_passed"    : all_passed,
        "max_errors"    : max_errors,
        "mean_max_error": mean_err,
    }


if __name__ == "__main__":
    # Quick self-test
    print("\n[equivariant_models.py] Running self-tests...\n")
    net = D4EquivariantNet(board_size=13, in_channels=3).eval()
    r1  = verify_policy_equivariance(net, verbose=True)
    r2  = verify_value_invariance(net,   verbose=True)
    assert r1["all_passed"], "Policy equivariance verification FAILED!"
    assert r2["all_passed"], "Value invariance verification FAILED!"
    print("\nAll self-tests passed. PASS\n")
