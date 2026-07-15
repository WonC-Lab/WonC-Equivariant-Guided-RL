"""
run_rigorous_validation.py ? Comprehensive Experimental Validation Suite
=========================================================================
Implements all experiments needed for AAMAS / IROS / ICRA / AAAI / IJCAI submission:

  1. Main comparison table  ? 5 methods ? 6 agent counts ? 1 layout, N=50 MCTS / N=100 fast
  2. Robustness analysis    ? PE-GNN+MCTS across 3 layouts, N=50
  3. Theorem 4 verification ? d_min vs. bound vs. success rate table
  4. Rc sensitivity         ? Rc ? {3, 6, ?} for M=4, N=50
  5. Grid-size transfer     ? 13?13 trained ? 20?20 zero-shot, M=4, N=50
  6. Hyperparameter sweeps  ? Nsearch ? {5,10,20,40,80} and ? ? {0,.1,.3,.5,1}
  7. Permutation equivariance verification (Theorem 1)

Usage:
  python run_rigorous_validation.py
"""

import os
import sys
import json
import math
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_agent_env import MultiAgentNavigationEnv
from equivariant_gnn import PermutationEquivariantGNN
from multi_agent_mcts import MultiAgentMCTS

# -------------------------------------------------------------------------
# Configuration
# Timing reference (CPU, M=4, Nsearch=20):
#   - 1 MCTS step  ~0.12s  =>  episode (max 80 steps) ~6-10s
#   - NUM_MCTS_EPISODES=20 x 6 M-values x 5 methods => ~30-40 min total
# Increase NUM_MCTS_EPISODES for publication; keep low for quick iteration.
# -------------------------------------------------------------------------
NUM_MCTS_EPISODES   = 20    # Episodes per condition for MCTS-based modes
NUM_FAST_EPISODES   = 50    # Episodes per condition for non-MCTS modes (fast)
FAST_MODE           = False  # Set True to skip slow sensitivity sweeps
AGENT_COUNTS        = [2, 3, 4, 5, 6, 8]
MCTS_SEARCHES_EVAL  = 20    # Reduced from 40 for speed; raise for camera-ready
HEURISTIC_BETA      = 0.3   # Heuristic mixing weight in MCTS during eval
GAMMA               = 0.95

# Theorem 4 bound parameters
THEOREM4_C          = 1.0
THEOREM4_ALPHA      = 2.0


# ?????????????????????????????????????????????????????????????????????????????
# Helper Utilities
# ?????????????????????????????????????????????????????????????????????????????

def compute_theorem4_bound(M, d_min_avg, gamma=GAMMA, C=THEOREM4_C, alpha=THEOREM4_ALPHA):
    """
    Computes the Equivariant Additive Value Decomposition error bound
    from Theorem 4:  ? ? C ? C(M,2) / ((1-?) ? d_min^?)
    """
    if d_min_avg <= 0 or d_min_avg == float('inf'):
        return float('inf')
    n_pairs = M * (M - 1) / 2
    return C * n_pairs / ((1 - gamma) * (d_min_avg ** alpha))


def gnn_lookahead_action(model, env, state, agent_idx):
    """
    1-step GNN lookahead planning for agent_idx.
    Tries each valid action, simulates 1 step (other agents use their heuristic
    greedy action), evaluates the resulting state with the GNN value head,
    and returns the action with the highest estimated value.
    Much stronger than pure greedy GNN; avoids the 0% strawman failure mode.
    """
    valid_actions = env.get_valid_actions(state, agent_idx)
    num_agents    = env.num_agents
    active_mask   = state[3]

    # Collect heuristic greedy action for every OTHER agent
    other_actions = []
    for j in range(num_agents):
        if j != agent_idx and active_mask[j]:
            heur = env.get_heuristic_policy(state, j)
            v_j  = env.get_valid_actions(state, j)
            masked = np.zeros(8)
            for a in v_j:
                masked[a] = heur[a]
            if masked.sum() > 0:
                other_actions.append(int(np.argmax(masked)))
            else:
                other_actions.append(0)
        else:
            other_actions.append(0)

    # Simulate each valid action and collect next-state observations (batched)
    obs_batch = []
    for action in valid_actions:
        joint = list(other_actions)
        joint[agent_idx] = action
        next_state, _, _, _ = env.step(state, tuple(joint))
        obs_batch.append(env.get_joint_observation(next_state).numpy())

    obs_tensor = torch.tensor(np.stack(obs_batch), dtype=torch.float32)  # (V, M, 3, H, W)
    model.eval()
    with torch.no_grad():
        _, values = model(obs_tensor)  # (V, M, 1)

    # Select action maximising agent_idx's estimated value
    best_action = valid_actions[0]
    best_value  = -float('inf')
    for i, action in enumerate(valid_actions):
        v = values[i, agent_idx].item()
        if v > best_value:
            best_value  = v
            best_action = action
    return best_action


def sample_random_starts_goals(num_agents, size, obstacles, rng):
    """Sample non-colliding random start/goal positions for all agents."""
    forbidden  = set(obstacles)
    candidates = [(r, c) for r in range(1, size - 1)
                          for c in range(1, size - 1)
                          if (r, c) not in forbidden]
    rng.shuffle(candidates)
    if len(candidates) < 2 * num_agents:
        return None, None
    starts = tuple(candidates[:num_agents])
    goals  = tuple(candidates[num_agents: 2 * num_agents])
    # Discard trivial episodes (start == goal)
    if any(starts[i] == goals[i] for i in range(num_agents)):
        return None, None
    return starts, goals


def sample_random_obstacles(size, count, starts, goals, rng):
    """Sample random obstacle positions, avoiding starts and goals."""
    forbidden  = set(starts) | set(goals)
    candidates = [(r, c) for r in range(size)
                          for c in range(size)
                          if (r, c) not in forbidden]
    rng.shuffle(candidates)
    return set(candidates[:count])


# ?????????????????????????????????????????????????????????????????????????????
# Core Evaluation Function
# ?????????????????????????????????????????????????????????????????????????????

def evaluate_performance(model, num_agents, obstacle_mode="default",
                         num_episodes=50, mode="mcts",
                         mcts_searches=MCTS_SEARCHES_EVAL,
                         obstacle_count=12, beta=HEURISTIC_BETA,
                         rc=None, grid_size=13):
    """
    Evaluates a single (method, condition) pair.

    Parameters
    ----------
    mode : str
        'mcts'          ? PE-GNN + MCTS (ours)
        'mcts_rc'       ? PE-GNN + MCTS with limited comm radius rc
        'gnn_lookahead' ? GNN 1-step value lookahead (new baseline)
        'gnn'           ? Pure GNN greedy (ablation)
        'heuristic'     ? Potential-field heuristic only (ablation)
        'orca'          ? Tuned ORCA-approximation (classical baseline)
    rc : float or None
        Communication radius in grid cells. None means unlimited (Rc=?).

    Returns
    -------
    dict with keys: success_rate, std, avg_d_min, theorem4_bound
    """
    env  = MultiAgentNavigationEnv(size=grid_size, num_agents=num_agents)
    mcts = MultiAgentMCTS(model=model, c_puct=1.4)

    episode_success = []
    episode_d_min   = []  # Mean d_min per episode (for Theorem 4)

    for ep in range(num_episodes):
        rng = random.Random(ep * 1000 + num_agents * 7 + hash(obstacle_mode) % 997)

        # --- Obstacle layout ---
        if obstacle_mode == "empty":
            env.obstacles = set()
        elif obstacle_mode in ("random", "density"):
            tmp_s = env.default_starts[:num_agents]
            tmp_g = env.default_goals[:num_agents]
            env.obstacles = sample_random_obstacles(
                grid_size, obstacle_count, tmp_s, tmp_g, rng)
        # "default" keeps env.obstacles from __init__

        # --- Randomise starts/goals ---
        starts, goals = sample_random_starts_goals(
            num_agents, grid_size, env.obstacles, rng)
        if starts is None:
            starts = tuple(env.default_starts[:num_agents])
            goals  = tuple(env.default_goals[:num_agents])

        env.starts = starts
        env.goals  = goals

        state    = env.generate_initial_state()
        done     = False
        step     = 0
        max_steps = num_agents * 20
        step_d_mins = []

        # ?? patch joint_observation with rc if needed ??????????????????????
        if rc is not None:
            _orig_joint_obs = env.get_joint_observation
            env.get_joint_observation = lambda s, **_: _orig_joint_obs(s, rc=rc)

        while not done and step < max_steps:
            joint_action = []
            active_mask  = state[3]

            # Track d_min for Theorem 4 verification
            d = MultiAgentNavigationEnv.compute_d_min(state)
            if d != float('inf'):
                step_d_mins.append(d)

            # ?? Action selection by mode ???????????????????????????????????
            if mode in ("mcts", "mcts_rc"):
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    _, probs = mcts.get_action_probabilities(
                        state, agent_idx=i, env=env,
                        num_searches=mcts_searches, temp=0.0, beta=beta)
                    joint_action.append(int(np.argmax(probs)))

            elif mode == "gnn_lookahead":
                model.eval()
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    joint_action.append(
                        gnn_lookahead_action(model, env, state, i))

            elif mode == "gnn":
                obs_joint = env.get_joint_observation(state)
                model.eval()
                with torch.no_grad():
                    logits, _ = model(obs_joint.unsqueeze(0))
                logits = logits.squeeze(0)
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    v_actions = env.get_valid_actions(state, i)
                    a_logits  = logits[i].clone()
                    inv = [a for a in range(8) if a not in v_actions]
                    a_logits[inv] = -1e9
                    joint_action.append(int(torch.argmax(a_logits).item()))

            elif mode == "heuristic":
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    hprobs   = env.get_heuristic_policy(state, i)
                    v_acts   = env.get_valid_actions(state, i)
                    masked_h = np.zeros(8)
                    for a in v_acts:
                        masked_h[a] = hprobs[a]
                    if masked_h.sum() > 0:
                        masked_h /= masked_h.sum()
                        joint_action.append(int(np.argmax(masked_h)))
                    else:
                        joint_action.append(0)

            elif mode == "orca":
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    oprobs   = env.get_orca_policy(state, i)
                    v_acts   = env.get_valid_actions(state, i)
                    masked_o = np.zeros(8)
                    for a in v_acts:
                        masked_o[a] = oprobs[a]
                    if masked_o.sum() > 0:
                        masked_o /= masked_o.sum()
                        joint_action.append(int(np.argmax(masked_o)))
                    else:
                        joint_action.append(0)

            state, _, done, _ = env.step(state, tuple(joint_action))
            step += 1

        # ?? Restore original joint_observation ????????????????????????????
        if rc is not None:
            env.get_joint_observation = _orig_joint_obs

        end_pos, end_goal, _, _ = state
        reached = sum(1 for i in range(num_agents)
                      if end_pos[i] == end_goal[i])
        episode_success.append(1 if reached == num_agents else 0)
        if step_d_mins:
            episode_d_min.append(float(np.mean(step_d_mins)))

    sr       = float(np.mean(episode_success))
    std      = float(np.std(episode_success))
    d_min_avg = float(np.mean(episode_d_min)) if episode_d_min else float('inf')
    bound     = compute_theorem4_bound(num_agents, d_min_avg)

    return {
        "success_rate": sr,
        "std": std,
        "avg_d_min": d_min_avg,
        "theorem4_bound": bound,
    }


# ?????????????????????????????????????????????????????????????????????????????
# Permutation Equivariance Check  (Theorem 1)
# ?????????????????????????????????????????????????????????????????????????????

def check_permutation_equivariance(model, num_agents=4, size=13):
    print("\n" + "-" * 55)
    print(" Theorem 1: Permutation Equivariance Verification")
    print("-" * 55)
    env   = MultiAgentNavigationEnv(size=size, num_agents=num_agents)
    state = env.generate_initial_state()
    obs   = env.get_joint_observation(state)

    model.eval()
    with torch.no_grad():
        orig_logits, _ = model(obs.unsqueeze(0))
    orig_logits = orig_logits.squeeze(0)

    diffs = []
    for _ in range(50):
        perm     = list(range(num_agents))
        random.shuffle(perm)
        perm_obs = obs[perm]
        with torch.no_grad():
            perm_logits, _ = model(perm_obs.unsqueeze(0))
        perm_logits = perm_logits.squeeze(0)
        expected    = orig_logits[perm]
        diffs.append(torch.max(torch.abs(perm_logits - expected)).item())

    max_d  = max(diffs)
    mean_d = float(np.mean(diffs))
    print(f"  Max  |f(Psigma Z) - Psigma f(Z)|inf : {max_d:.2e}")
    print(f"  Mean |f(Psigma Z) - Psigma f(Z)|inf : {mean_d:.2e}")
    status = "[OK] CONFIRMED" if max_d < 1e-4 else "[WARN] High numerical differences detected"
    print(f"  Status: {status}  (threshold 1e-4, IEEE 754 noise expected)")
    print("-" * 55 + "\n")
    return max_d


# -----------------------------------------------------------------------------
# Experiment 1: Main Comparison Table
# ?????????????????????????????????????????????????????????????????????????????

def run_main_comparison(model):
    """
    5-method comparison on Default Map across all agent counts.
    Returns dict keyed by method name ? list of result-dicts per M.
    """
    print("\n" + "=" * 55)
    print("  Exp 1 -- Main Comparison (Default Map)")
    print("=" * 55)

    methods = [
        ("mcts",          "PE-GNN + MCTS (Ours)",        NUM_MCTS_EPISODES),
        ("orca",          "ORCA-approx. (Classical)",     NUM_FAST_EPISODES),
        ("gnn_lookahead", "GNN + 1-step Lookahead",       NUM_FAST_EPISODES),
        ("gnn",           "PE-GNN Only (No Search)",      NUM_FAST_EPISODES),
        ("heuristic",     "Heuristic Only",               NUM_FAST_EPISODES),
    ]

    comparison_results = {}
    for mode, label, n_ep in methods:
        comparison_results[mode] = []
        print(f"\n  Method: {label}  (N={n_ep})")
        for M in AGENT_COUNTS:
            res = evaluate_performance(
                model, num_agents=M, obstacle_mode="default",
                num_episodes=n_ep, mode=mode)
            comparison_results[mode].append(res)
            sr  = res["success_rate"] * 100
            std = res["std"] * 100
            print(f"    M={M}: {sr:.1f}% ? {std:.1f}%  "
                  f"[d_min={res['avg_d_min']:.2f}  "
                  f"Thm4_bound={res['theorem4_bound']:.3f}]")

    return comparison_results


# ?????????????????????????????????????????????????????????????????????????????
# Experiment 2: Robustness across Obstacle Layouts
# ?????????????????????????????????????????????????????????????????????????????

def run_robustness(model):
    """PE-GNN+MCTS across 3 obstacle layouts, all agent counts."""
    print("\n" + "=" * 55)
    print("  Exp 2 -- Robustness across Obstacle Layouts")
    print("=" * 55)

    layouts = ["default", "empty", "random"]
    robustness_results = {}
    for layout in layouts:
        robustness_results[layout] = []
        print(f"\n  Layout: {layout.upper()}")
        for M in AGENT_COUNTS:
            res = evaluate_performance(
                model, num_agents=M, obstacle_mode=layout,
                num_episodes=NUM_MCTS_EPISODES, mode="mcts")
            robustness_results[layout].append(res)
            sr  = res["success_rate"] * 100
            std = res["std"] * 100
            print(f"    M={M}: {sr:.1f}% ? {std:.1f}%")

    return robustness_results


# ?????????????????????????????????????????????????????????????????????????????
# Experiment 3: Theorem 4 Quantitative Verification
# ?????????????????????????????????????????????????????????????????????????????

def run_theorem4_verification(model, comparison_results):
    """
    Builds the Theorem 4 consistency table from already-computed comparison_results
    (MCTS method). Prints and returns the table.
    """
    print("\n" + "=" * 55)
    print("  Exp 3 -- Theorem 4 Quantitative Verification")
    print("=" * 55)
    print(f"\n  {'M':>4}  {'avg d_min':>10}  {'Thm4 Bound':>12}  "
          f"{'Success':>9}  {'1-Success':>10}")
    print("  " + "-" * 52)

    table = []
    for i, M in enumerate(AGENT_COUNTS):
        res     = comparison_results["mcts"][i]
        d_min   = res["avg_d_min"]
        bound   = res["theorem4_bound"]
        sr      = res["success_rate"]
        failure = 1.0 - sr
        print(f"  {M:>4}  {d_min:>10.3f}  {bound:>12.4f}  "
              f"{sr*100:>8.1f}%  {failure:>10.4f}")
        table.append({"M": M, "avg_d_min": d_min, "bound": bound,
                      "success_rate": sr, "failure": failure})
    return table


# ?????????????????????????????????????????????????????????????????????????????
# Experiment 4: Communication Radius Sensitivity
# ?????????????????????????????????????????????????????????????????????????????

def run_rc_sensitivity(model):
    """
    Tests sensitivity to communication radius Rc ? {3, 6, ?} for M=4, Default Map.
    Rc limits what each agent can see of other agents (observation masking).
    """
    print("\n" + "=" * 55)
    print("  Exp 4 -- Communication Radius Sensitivity (M=4)")
    print("=" * 55)

    rc_values = [3, 6, None]  # None = ?
    rc_labels = ["Rc = 3 cells", "Rc = 6 cells", "Rc = ? (full)"]
    rc_results = []

    for rc, label in zip(rc_values, rc_labels):
        res = evaluate_performance(
            model, num_agents=4, obstacle_mode="default",
            num_episodes=NUM_MCTS_EPISODES, mode="mcts", rc=rc)
        rc_results.append(res)
        sr  = res["success_rate"] * 100
        std = res["std"] * 100
        drop = ""
        if rc is not None:
            full_sr = None  # filled after loop
        print(f"    {label}: {sr:.1f}% ? {std:.1f}%")

    # Compute degradation vs. Rc=?
    full_sr = rc_results[-1]["success_rate"]
    print(f"\n  Degradation vs. Rc=?:")
    for rc, label, res in zip(rc_values, rc_labels, rc_results):
        drop = (full_sr - res["success_rate"]) * 100
        print(f"    {label}: ? = {-drop:+.1f}%")

    return {"rc_values": rc_values, "rc_labels": rc_labels, "results": rc_results}


# ?????????????????????????????????????????????????????????????????????????????
# Experiment 5: 20?20 Grid Zero-Shot Transfer
# ?????????????????????????????????????????????????????????????????????????????

def run_grid_transfer(model):
    """
    Zero-shot transfer from the 13?13 trained model to 20?20 grid.
    Model generalises because CNN uses AdaptiveAvgPool2d (size-agnostic).
    """
    print("\n" + "=" * 55)
    print("  Exp 5 -- Grid-Size Zero-Shot Transfer (M=4)")
    print("=" * 55)

    grid_results = {}
    for size, label in [(13, "13?13 (trained)"), (20, "20?20 (zero-shot)")]:
        res = evaluate_performance(
            model, num_agents=4, obstacle_mode="random",
            num_episodes=NUM_MCTS_EPISODES, mode="mcts", grid_size=size)
        grid_results[size] = res
        sr  = res["success_rate"] * 100
        std = res["std"] * 100
        print(f"    {label}: {sr:.1f}% ? {std:.1f}%")

    return grid_results


# ?????????????????????????????????????????????????????????????????????????????
# Experiment 6: Hyperparameter Sensitivity (Nsearch & ?)
# ?????????????????????????????????????????????????????????????????????????????

def run_sensitivity_experiments(model):
    """Nsearch and ? sweeps for M=4, Default Map."""
    print("\n" + "=" * 55)
    print("  Exp 6 -- Hyperparameter Sensitivity (M=4)")
    print("=" * 55)

    # Nsearch sweep
    print("\n  a) MCTS Search Budget Nsearch sweep (?=0.3 fixed)")
    nsearch_vals = [5, 10, 20, 40, 80]
    nsearch_results = []
    for ns in nsearch_vals:
        res = evaluate_performance(
            model, num_agents=4, obstacle_mode="default",
            num_episodes=NUM_MCTS_EPISODES, mode="mcts",
            mcts_searches=ns, beta=0.3)
        nsearch_results.append(res)
        print(f"    Nsearch={ns:3d}: {res['success_rate']*100:.1f}% ? {res['std']*100:.1f}%")

    # ? sweep
    print("\n  b) Heuristic Mixing Weight ? sweep (Nsearch=40 fixed)")
    beta_vals = [0.0, 0.1, 0.3, 0.5, 1.0]
    beta_results = []
    for b in beta_vals:
        res = evaluate_performance(
            model, num_agents=4, obstacle_mode="default",
            num_episodes=NUM_MCTS_EPISODES, mode="mcts",
            mcts_searches=MCTS_SEARCHES_EVAL, beta=b)
        beta_results.append(res)
        print(f"    ?={b:.1f}: {res['success_rate']*100:.1f}% ? {res['std']*100:.1f}%")

    return {
        "nsearch": {"values": nsearch_vals, "results": nsearch_results},
        "beta":    {"values": beta_vals,    "results": beta_results},
    }


# ?????????????????????????????????????????????????????????????????????????????
# Plotting
# ?????????????????????????????????????????????????????????????????????????????

COLORS  = {
    "mcts":          "#1f77b4",
    "orca":          "#9467bd",
    "gnn_lookahead": "#8c564b",
    "heuristic":     "#2ca02c",
    "gnn":           "#d62728",
    "default":       "#1f77b4",
    "empty":         "#2ca02c",
    "random":        "#d62728",
}
MARKERS = {"mcts": "o", "orca": "^", "gnn_lookahead": "D",
           "heuristic": "s", "gnn": "x",
           "default": "o", "empty": "s", "random": "d"}
LINE_STYLES = {"mcts": "-", "orca": "--", "gnn_lookahead": "-.",
               "heuristic": ":", "gnn": "--",
               "default": "-", "empty": "--", "random": "-."}


def _apply_style(ax):
    ax.set_xticks(AGENT_COUNTS)
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.axvline(x=4, color="black", linestyle=":", alpha=0.5,
               label="Training limit (M=4)")
    ax.set_xlabel("Number of Cooperative Agents  $M$", fontsize=11)
    ax.set_ylabel("Coordination Success Rate (%)\n"
                  f"mean ? std  ({NUM_MCTS_EPISODES} episodes)", fontsize=10)


def plot_main_comparison(comparison_results):
    """Figure: all 5 methods on Default Map."""
    labels = {
        "mcts":          "PE-GNN + MCTS (Ours)",
        "orca":          "ORCA-approx. (tuned)",
        "gnn_lookahead": "GNN + 1-step Lookahead",
        "heuristic":     "Heuristic Only",
        "gnn":           "PE-GNN Only",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in labels.items():
        rates = np.array([r["success_rate"] for r in comparison_results[key]]) * 100
        stds  = np.array([r["std"]          for r in comparison_results[key]]) * 100
        n_ep  = NUM_MCTS_EPISODES if key == "mcts" else NUM_FAST_EPISODES
        ax.plot(AGENT_COUNTS, rates,
                marker=MARKERS[key], linestyle=LINE_STYLES[key],
                color=COLORS[key], linewidth=2.2,
                label=f"{label}  (N={n_ep})")
        ax.fill_between(AGENT_COUNTS, rates - stds, rates + stds,
                        alpha=0.12, color=COLORS[key])
    _apply_style(ax)
    ax.set_title("Main Comparison: Success Rate vs. Agent Count  (Default Map)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig("results/mcts_ablation.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/mcts_ablation.png")


def plot_robustness(robustness_results):
    """Figure: PE-GNN+MCTS across 3 layouts."""
    layout_labels = {
        "default": "Default Map (Trained Layout)",
        "empty":   "Empty Map (No Obstacles)",
        "random":  "Random Obstacles (Unseen)",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in layout_labels.items():
        rates = np.array([r["success_rate"] for r in robustness_results[key]]) * 100
        stds  = np.array([r["std"]          for r in robustness_results[key]]) * 100
        ax.plot(AGENT_COUNTS, rates,
                marker=MARKERS[key], linestyle=LINE_STYLES[key],
                color=COLORS[key], linewidth=2.2, label=label)
        ax.fill_between(AGENT_COUNTS, rates - stds, rates + stds,
                        alpha=0.12, color=COLORS[key])
    _apply_style(ax)
    ax.set_title("Robustness: PE-GNN+MCTS across Obstacle Configurations",
                 fontsize=12, fontweight="bold", pad=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("results/scalability_test.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/scalability_test.png")


def plot_theorem4(theorem4_table):
    """Dual-axis figure: d_min (left) and Theorem 4 bound / failure rate (right)."""
    Ms       = [row["M"]        for row in theorem4_table]
    d_mins   = [row["avg_d_min"] for row in theorem4_table]
    bounds   = [min(row["bound"], 5.0) for row in theorem4_table]  # cap for viz
    failures = [row["failure"]  for row in theorem4_table]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(Ms, d_mins, "o-",  color="#1f77b4", linewidth=2.2, label="avg $d_{\\rm min}$ (cells)")
    l2, = ax2.plot(Ms, bounds, "s--", color="#9467bd", linewidth=2.2, label="Theorem 4 bound (capped 5)")
    l3, = ax2.plot(Ms, failures, "D-.", color="#d62728", linewidth=2.2, label="Failure rate $1 - \\text{SR}$")

    ax1.set_xlabel("Number of Agents  $M$", fontsize=11)
    ax1.set_ylabel("Average $d_{\\rm min}$ (grid cells)", fontsize=11, color="#1f77b4")
    ax2.set_ylabel("Bound / Failure Rate", fontsize=11, color="#555")
    ax1.set_xticks(Ms)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title("Theorem 4 Consistency: $d_{\\rm min}$ vs. Decomposition Error Bound vs. Failure",
                  fontsize=11, fontweight="bold", pad=12)
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/theorem4_verification.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/theorem4_verification.png")


def plot_rc_sensitivity(rc_data):
    """Bar chart: Rc ? {3, 6, ?} success rates for M=4."""
    labels = ["$R_c=3$", "$R_c=6$", "$R_c=\\infty$"]
    rates  = [r["success_rate"] * 100 for r in rc_data["results"]]
    stds   = [r["std"]          * 100 for r in rc_data["results"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x      = np.arange(len(labels))
    bars   = ax.bar(x, rates, width=0.5,
                    color=["#d62728", "#ff7f0e", "#1f77b4"],
                    alpha=0.85, yerr=stds, capsize=6,
                    error_kw=dict(elinewidth=1.5, ecolor="black"))
    for idx, (r, s) in enumerate(zip(rates, stds)):
        ax.text(idx, r + s + 2, f"{r:.1f}%", ha="center",
                fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Success Rate (%)  mean ? std", fontsize=11)
    ax.set_title(f"Communication Radius Sensitivity  ($M=4$, Default Map, $N={NUM_MCTS_EPISODES}$)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    fig.tight_layout()
    fig.savefig("results/rc_sensitivity.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/rc_sensitivity.png")


def plot_grid_transfer(grid_results):
    """Bar chart: 13?13 vs 20?20 zero-shot."""
    sizes  = [13, 20]
    labels = ["13?13\n(trained)", "20?20\n(zero-shot)"]
    rates  = [grid_results[s]["success_rate"] * 100 for s in sizes]
    stds   = [grid_results[s]["std"]          * 100 for s in sizes]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    x = np.arange(2)
    ax.bar(x, rates, width=0.45, color=["#1f77b4", "#2ca02c"],
           alpha=0.85, yerr=stds, capsize=8,
           error_kw=dict(elinewidth=1.5, ecolor="black"))
    for idx, (r, s) in enumerate(zip(rates, stds)):
        ax.text(idx, r + s + 2, f"{r:.1f}%", ha="center",
                fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Success Rate (%)  mean ? std", fontsize=11)
    ax.set_title(f"Zero-Shot Grid-Size Transfer  ($M=4$, Random Obs., $N={NUM_MCTS_EPISODES}$)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    fig.tight_layout()
    fig.savefig("results/grid_transfer.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/grid_transfer.png")


def plot_sensitivity(sens_data):
    """Nsearch and ? sensitivity figures."""
    # Nsearch
    ns_vals   = sens_data["nsearch"]["values"]
    ns_rates  = [r["success_rate"] * 100 for r in sens_data["nsearch"]["results"]]
    ns_stds   = [r["std"]          * 100 for r in sens_data["nsearch"]["results"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(ns_vals, ns_rates, yerr=ns_stds, marker="o", linewidth=2.2,
                color="#1f77b4", capsize=5, elinewidth=1.5, label="Success Rate")
    ax.fill_between(ns_vals,
                    [r - s for r, s in zip(ns_rates, ns_stds)],
                    [r + s for r, s in zip(ns_rates, ns_stds)],
                    alpha=0.15, color="#1f77b4")
    ax.set_xlabel("MCTS Search Budget  $N_{\\rm search}$", fontsize=11)
    ax.set_ylabel(f"Success Rate (%)  N={NUM_MCTS_EPISODES}", fontsize=11)
    ax.set_title("Sensitivity to MCTS Budget  ($M=4$, $\\beta=0.3$, Default Map)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(-5, 110)
    ax.grid(True, linestyle="--", alpha=0.55)
    fig.tight_layout()
    fig.savefig("results/nsearch_sensitivity.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/nsearch_sensitivity.png")

    # ?
    b_vals   = sens_data["beta"]["values"]
    b_rates  = [r["success_rate"] * 100 for r in sens_data["beta"]["results"]]
    b_stds   = [r["std"]          * 100 for r in sens_data["beta"]["results"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(b_vals, b_rates, yerr=b_stds, marker="s", linewidth=2.2,
                color="#ff7f0e", capsize=5, elinewidth=1.5, label="Success Rate")
    ax.fill_between(b_vals,
                    [r - s for r, s in zip(b_rates, b_stds)],
                    [r + s for r, s in zip(b_rates, b_stds)],
                    alpha=0.15, color="#ff7f0e")
    ax.set_xlabel("Heuristic Mixing Weight  $\\beta$", fontsize=11)
    ax.set_ylabel(f"Success Rate (%)  N={NUM_MCTS_EPISODES}", fontsize=11)
    ax.set_title("Sensitivity to $\\beta$  ($M=4$, $N_{\\rm search}=40$, Default Map)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(-5, 110)
    ax.grid(True, linestyle="--", alpha=0.55)
    fig.tight_layout()
    fig.savefig("results/beta_sensitivity.png", dpi=300)
    plt.close(fig)
    print("  ? Saved results/beta_sensitivity.png")


# ?????????????????????????????????????????????????????????????????????????????
# Orchestrator
# ?????????????????????????????????????????????????????????????????????????????

def run_rigorous_validation():
    print("=" * 56)
    print("  Rigorous Multi-Agent Validation Suite")
    print("  Target: AAMAS / IROS / ICRA / AAAI / IJCAI")
    print("=" * 56)
    print(f"\n  MCTS episodes / condition : {NUM_MCTS_EPISODES}")
    print(f"  Fast-mode episodes / cond : {NUM_FAST_EPISODES}")
    print(f"  MCTS searches (eval)      : {MCTS_SEARCHES_EVAL}")
    print(f"  Agent counts              : {AGENT_COUNTS}\n")

    os.makedirs("results", exist_ok=True)

    # ?? Load model ?????????????????????????????????????????????????????????
    model = PermutationEquivariantGNN(grid_size=13, in_channels=3, d_model=128)
    model_path = "models/multi_agent_model.pth"
    if not os.path.exists(model_path):
        print(f"  [!] Model not found at {model_path}. "
              "Run train_multi_agent.py first.\n  Aborting.")
        return
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"  Loaded model from: {model_path}\n")

    # ?? Theorem 1: Equivariance ?????????????????????????????????????????????
    check_permutation_equivariance(model)

    # ?? Experiment 1: Main Comparison ??????????????????????????????????????
    comparison_results = run_main_comparison(model)
    plot_main_comparison(comparison_results)

    # ?? Experiment 2: Robustness ????????????????????????????????????????????
    robustness_results = run_robustness(model)
    plot_robustness(robustness_results)

    # ?? Experiment 3: Theorem 4 Verification ???????????????????????????????
    theorem4_table = run_theorem4_verification(model, comparison_results)
    plot_theorem4(theorem4_table)

    # ?? Experiment 4: Rc Sensitivity ????????????????????????????????????????
    rc_data = run_rc_sensitivity(model)
    plot_rc_sensitivity(rc_data)

    # ?? Experiment 5: Grid Transfer ?????????????????????????????????????????
    grid_results = run_grid_transfer(model)
    plot_grid_transfer(grid_results)

    # -- Experiment 6: Hyperparameter Sensitivity (optional, slow) -----------
    all_results = {
        "config": {
            "num_mcts_episodes": NUM_MCTS_EPISODES,
            "num_fast_episodes": NUM_FAST_EPISODES,
            "agent_counts":      AGENT_COUNTS,
            "mcts_searches":     MCTS_SEARCHES_EVAL,
            "beta":              HEURISTIC_BETA,
            "theorem4_C":        THEOREM4_C,
            "theorem4_alpha":    THEOREM4_ALPHA,
        },
        "comparison":   {k: v for k, v in comparison_results.items()},
        "robustness":   {k: v for k, v in robustness_results.items()},
        "theorem4":     theorem4_table,
        "rc_sensitivity": {
            "rc_values": [str(r) for r in rc_data["rc_values"]],
            "results":   rc_data["results"],
        },
        "grid_transfer": {str(k): v for k, v in grid_results.items()},
        "sensitivity": {
            "nsearch": {
                "values":  sens_data["nsearch"]["values"],
                "results": sens_data["nsearch"]["results"],
            },
            "beta": {
                "values":  sens_data["beta"]["values"],
                "results": sens_data["beta"]["results"],
            },
        },
    }

    json_path = "results/academic_results_rigorous.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4, default=str)

    print("\n" + "=" * 56)
    print("  Validation Complete -- Output Files")
    print("=" * 56)
    print("  results/mcts_ablation.png       (main comparison)")
    print("  results/scalability_test.png    (robustness)")
    print("  results/theorem4_verification.png")
    print("  results/rc_sensitivity.png")
    print("  results/grid_transfer.png")
    print("  results/nsearch_sensitivity.png")
    print("  results/beta_sensitivity.png")
    print("  results/academic_results_rigorous.json")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    run_rigorous_validation()
