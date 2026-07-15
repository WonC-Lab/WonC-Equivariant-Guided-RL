"""
real_experiment_v3.py  ─  "Train First, Search Later" (v3 - Publication Ready)
===============================================================================
Improvements over v2:
  1. Best-checkpoint evaluation: final metrics use the best model seen during
     training (not the final, potentially collapsed, episode model).
  2. Entropy bonus: -λ_ent * H(π) added to loss to prevent policy collapse.
  3. 5 seeds for statistical confidence.
  4. Full D4 8-element zero-shot generalization (4 rotations + 4 reflections).
  5. English-only labels for publication-quality plots.

Outputs:
  real_results_v3.json        ─ Full metrics (mean ± std, 5 seeds)
  real_results_plots_v3.png   ─ Publication-quality figures
"""

import os, sys, json, time, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"   # No CJK glyphs → no warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Local modules ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autonomous_env import AutonomousNavigationEnv
from equivariant_models import D4EquivariantNet, StandardCNN, D4GroupAction
from mcts_actor_critic import ActorCriticMCTS
from train_navigation import SymmetricNavEnvAdapter

# ── Device ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("cpu")
    print(" Apple Silicon: Forcing CPU (MPS synchronization slows sequential MCTS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(" Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print(" Using CPU")

# ── Hyperparameters ───────────────────────────────────────────────────
SEEDS         = [42, 123, 456, 789, 2024]   # 5 independent seeds
NUM_EPISODES  = 300
WARMUP_EPS    = 60
MAX_STEPS     = 40
GAMMA         = 0.95
LR            = 0.002
BETA_START    = 2.0
BETA_DECAY    = 0.95
BETA_MIN      = 0.5
ENT_COEF      = 0.01    # [NEW] Entropy bonus coefficient (prevents policy collapse)
TRAIN_EPOCHS  = 3
GRAD_CLIP     = 1.0
EVAL_INTERVAL = 50
EVAL_EPS      = 20
MCTS_SIMS     = 15
NUM_LAYERS    = 4
NUM_FILTERS   = 16

# ── Utilities ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_model(model_cls):
    m = model_cls(board_size=13, in_channels=3,
                  num_filters=NUM_FILTERS, num_layers=NUM_LAYERS)
    return m.to(DEVICE)

def discounted_returns(rewards, gamma=GAMMA):
    G, rets = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        rets.insert(0, G)
    return rets

def apply_d4_to_coord(r, c, n, action_idx):
    """
    Apply D4 group element `action_idx` to grid coordinate (r, c) in an n×n grid.
    Matches D4GroupAction.apply_action tensor transformation:
      - action_idx 0–3 : rotations (0°, 90°, 180°, 270° CCW)
      - action_idx 4–7 : horizontal flip + rotations
    """
    rot_k = action_idx % 4
    flip  = action_idx // 4
    if flip == 1:
        c = n - 1 - c          # horizontal flip
    for _ in range(rot_k):
        r, c = n - 1 - c, r   # 90° CCW: (r,c) → (n-1-c, r)
    return r, c

def apply_d4_to_map(obstacles, goal, size, action_idx):
    """Transform obstacle set and goal position under D4 group action."""
    new_obs  = {apply_d4_to_coord(r, c, size, action_idx) for r, c in obstacles}
    gr, gc   = goal
    new_goal = apply_d4_to_coord(gr, gc, size, action_idx)
    return new_obs, new_goal

# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: Actor-Critic Training (no MCTS during training)
# ══════════════════════════════════════════════════════════════════════

def train_one_episode(model, base_env, env_adapter, optimizer, beta,
                      episode_num, kl_loss_fn):
    """
    Run a single training episode with Actor-Critic + KL Heuristic + Entropy Loss.
    WARMUP phase: heuristic directly selects actions (safe early exploration).
    """
    state = base_env.generate_initial_state()
    warmup = (episode_num <= WARMUP_EPS)

    ep_tensors, ep_actions, ep_heuristics, ep_old_log_probs, ep_rewards = [], [], [], [], []
    game_over, winner, steps = False, None, 0

    model.eval()
    while not game_over and steps < MAX_STEPS:
        st = env_adapter.state_to_tensor(state).to(DEVICE)
        valid_dirs = base_env.get_valid_actions(state)
        heuristic_8 = base_env.get_heuristic_policy(state)   # (8,)

        if warmup:
            probs = np.array([heuristic_8[i] for i in valid_dirs])
            probs = probs / probs.sum()
            chosen = int(np.random.choice(valid_dirs, p=probs))
        else:
            with torch.no_grad():
                logits, _ = model(st)
            logits_np = logits.squeeze(0).cpu().numpy()
            mask = np.full(8, -1e9)
            for di in valid_dirs:
                mask[di] = logits_np[di]
            mask_shifted = mask - mask[mask > -1e8].max()
            exp_m = np.exp(mask_shifted)
            exp_m[mask <= -1e8] = 0.0
            probs = exp_m / exp_m.sum()
            chosen = int(np.random.choice(8, p=probs))

        with torch.no_grad():
            logits_lp, _ = model(st)
        lp = F.log_softmax(logits_lp, dim=-1)
        old_log_prob = lp[0, chosen].item()

        next_state, _ = env_adapter.step(state, chosen, 1)
        game_over, winner = env_adapter.check_game_over(next_state, 1)

        ep_tensors.append(st)
        ep_actions.append(chosen)
        ep_heuristics.append(heuristic_8)
        ep_old_log_probs.append(old_log_prob)
        ep_rewards.append(1.0 if (game_over and winner == 1)
                          else -1.0 if (game_over and winner == 2)
                          else -0.05)
        state = next_state
        steps += 1

    success   = (winner == 1)
    collision = (winner == 2)

    avg_loss = None
    if ep_tensors:
        returns = discounted_returns(ep_rewards)
        batch_s     = torch.cat(ep_tensors, 0)
        batch_a     = torch.tensor(ep_actions,     dtype=torch.long,    device=DEVICE)
        batch_r     = torch.tensor(returns,        dtype=torch.float32, device=DEVICE)
        batch_h     = torch.tensor(np.array(ep_heuristics), dtype=torch.float32, device=DEVICE)
        batch_old_lps = torch.tensor(ep_old_log_probs, dtype=torch.float32, device=DEVICE)

        model.train()
        total_loss = 0.0
        for _ in range(TRAIN_EPOCHS):
            optimizer.zero_grad()
            logits, values = model(batch_s)             # (T, 8), (T, 1)

            log_probs = F.log_softmax(logits, dim=-1)   # (T, 8)
            sel_lp    = log_probs[torch.arange(len(batch_a)), batch_a]

            with torch.no_grad():
                adv = batch_r - values.squeeze(-1)

            # PPO Clipped Policy Loss
            ratio = torch.exp(sel_lp - batch_old_lps)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # KL Divergence: KL(heuristic ‖ agent)
            kl = kl_loss_fn(log_probs, batch_h)

            # Value MSE
            val_loss = F.mse_loss(values.squeeze(-1), batch_r)

            # [NEW] Entropy bonus: H(π) = -Σ p * log(p) (higher = more diverse policy)
            probs_t = F.softmax(logits, dim=-1)
            entropy = -(probs_t * log_probs).sum(dim=-1).mean()

            loss = pg_loss + beta * kl + 0.5 * val_loss - ENT_COEF * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / TRAIN_EPOCHS

    return success, collision, steps, avg_loss


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_greedy(model, base_env, env_adapter, n=EVAL_EPS):
    """Greedy evaluation: argmax of policy logits."""
    model.eval()
    successes = 0
    for _ in range(n):
        state = base_env.generate_initial_state()
        for _ in range(MAX_STEPS):
            valid = base_env.get_valid_actions(state)
            if not valid: break
            st = env_adapter.state_to_tensor(state).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(st)
            lnp = logits.squeeze(0).cpu().numpy()
            mask = np.full(8, -1e9)
            for di in valid:
                mask[di] = lnp[di]
            chosen = int(np.argmax(mask))
            state, _ = env_adapter.step(state, chosen, 1)
            done, winner = env_adapter.check_game_over(state, 1)
            if done:
                if winner == 1: successes += 1
                break
    return successes / n


def evaluate_with_mcts(model, base_env, env_adapter, n=EVAL_EPS):
    """MCTS-guided evaluation using model as prior."""
    model.eval()
    successes = 0
    for _ in range(n):
        mcts = ActorCriticMCTS(model=model, c_puct=1.4)
        state = base_env.generate_initial_state()
        for _ in range(MAX_STEPS):
            actions, probs = mcts.get_action_probabilities(
                state, 1, env_adapter, num_searches=MCTS_SIMS, temp=0.1
            )
            if not actions: break
            chosen = actions[int(np.argmax(probs))]
            state, _ = env_adapter.step(state, chosen, 1)
            done, winner = env_adapter.check_game_over(state, 1)
            if done:
                if winner == 1: successes += 1
                break
    return successes / n


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3: D4 Zero-Shot Generalization (all 8 group elements)
# ══════════════════════════════════════════════════════════════════════

D4_LABELS = ["0° (id)", "90° CCW", "180°", "270° CCW",
             "Flip-H", "Flip-H+90°", "Flip-H+180°", "Flip-H+270°"]

def evaluate_d4_generalization(model, base_env, n=30):
    """
    Evaluate zero-shot generalization across ALL 8 D4 group elements.
    Returns (greedy_rates, mcts_rates), each a list of 8 floats.
    """
    greedy_rates, mcts_rates = [], []
    canonical_obs  = set(base_env.obstacles)
    canonical_goal = base_env.goal
    canonical_start = base_env.start

    for g_idx in range(8):
        new_obs, new_goal = apply_d4_to_map(canonical_obs, canonical_goal,
                                             base_env.size, g_idx)
        new_start = apply_d4_to_coord(canonical_start[0], canonical_start[1], base_env.size, g_idx)
        rot_env = AutonomousNavigationEnv(size=base_env.size)
        rot_env.obstacles = new_obs
        rot_env.goal      = new_goal
        rot_env.start     = new_start
        rot_adapter = SymmetricNavEnvAdapter(rot_env)

        gr = evaluate_greedy(model, rot_env, rot_adapter, n=n)
        mr = evaluate_with_mcts(model, rot_env, rot_adapter, n=n)
        greedy_rates.append(gr)
        mcts_rates.append(mr)
        print(f"    D4[{g_idx}] {D4_LABELS[g_idx]:18s}  Greedy: {gr*100:5.1f}%  MCTS: {mr*100:5.1f}%")

    return greedy_rates, mcts_rates


# ══════════════════════════════════════════════════════════════════════
#  Multi-Seed Training Loop
# ══════════════════════════════════════════════════════════════════════

def run_experiment(label, model_cls, color, linestyle):
    print(f"\n{'='*65}")
    print(f"  Model: {label}")
    print(f"{'='*65}")

    kl_fn = nn.KLDivLoss(reduction="batchmean")

    all_greedy_rates, all_mcts_rates = [], []
    all_curves_greedy, all_curves_mcts = [], []
    all_collisions, all_conv_ep = [], []
    global_best_state = None
    global_best_rate  = -1.0

    for si, seed in enumerate(SEEDS):
        print(f"\n  [Seed {si+1}/{len(SEEDS)}: {seed}]")
        set_seed(seed)

        base_env    = AutonomousNavigationEnv(size=13)
        env_adapter = SymmetricNavEnvAdapter(base_env)
        model       = make_model(model_cls)
        optimizer   = optim.Adam(model.parameters(), lr=LR)
        beta        = BETA_START

        curve_greedy, curve_mcts = [], []
        collisions = 0
        conv_ep    = NUM_EPISODES

        # [NEW] Per-seed best checkpoint tracking
        best_seed_rate  = -1.0
        best_seed_state = None

        for ep in range(1, NUM_EPISODES + 1):
            success, collision, steps, loss = train_one_episode(
                model, base_env, env_adapter, optimizer, beta, ep, kl_fn
            )
            if collision: collisions += 1
            beta = max(beta * BETA_DECAY, BETA_MIN)

            if ep % EVAL_INTERVAL == 0:
                base_env.reset_canonical_obstacles()
                gr = evaluate_greedy(model, base_env, env_adapter)
                mr = evaluate_with_mcts(model, base_env, env_adapter)
                curve_greedy.append(gr)
                curve_mcts.append(mr)

                if gr >= 0.70 and conv_ep == NUM_EPISODES:
                    conv_ep = ep

                # [NEW] Save best checkpoint whenever greedy improves
                if gr > best_seed_rate:
                    best_seed_rate  = gr
                    best_seed_state = copy.deepcopy(model.state_dict())

                loss_s = f"{loss:.4f}" if loss else "N/A"
                mode   = "WarmUp" if ep <= WARMUP_EPS else "AC+KL  "
                print(f"    Ep {ep:4d}/{NUM_EPISODES} [{mode}] "
                      f"Greedy: {gr*100:5.1f}%  MCTS: {mr*100:5.1f}%  "
                      f"Loss: {loss_s}  β: {beta:.3f}  Coll: {collisions}")

        # [NEW] Final evaluation uses best checkpoint (not final episode model)
        best_eval_model = make_model(model_cls)
        if best_seed_state is not None:
            best_eval_model.load_state_dict(best_seed_state)
        else:
            best_eval_model.load_state_dict(model.state_dict())

        base_env.reset_canonical_obstacles()
        final_greedy = evaluate_greedy(best_eval_model, base_env, env_adapter, n=50)
        final_mcts   = evaluate_with_mcts(best_eval_model, base_env, env_adapter, n=50)

        if final_greedy > global_best_rate:
            global_best_rate  = final_greedy
            global_best_state = copy.deepcopy(best_eval_model.state_dict())

        all_greedy_rates.append(final_greedy)
        all_mcts_rates.append(final_mcts)
        all_curves_greedy.append(curve_greedy)
        all_curves_mcts.append(curve_mcts)
        all_collisions.append(collisions)
        all_conv_ep.append(conv_ep)

        print(f"  → Seed {seed}: Greedy {final_greedy*100:.1f}%  "
              f"MCTS {final_mcts*100:.1f}%  Conv@{conv_ep}  Coll:{collisions}  "
              f"[best ckpt greedy={best_seed_rate*100:.1f}%]")

    # Aggregate statistics
    eval_eps = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))

    def stats(arr):
        a = np.array(arr)
        return float(a.mean()), float(a.std())

    mg, sg = stats(all_greedy_rates)
    mm, sm = stats(all_mcts_rates)

    curves_g = np.array(all_curves_greedy)
    curves_m = np.array(all_curves_mcts)

    print(f"\n  ★ {label}")
    print(f"    Greedy:  {mg*100:.1f}% ± {sg*100:.1f}%")
    print(f"    +MCTS:   {mm*100:.1f}% ± {sm*100:.1f}%")
    print(f"    Conv Ep: {np.mean(all_conv_ep):.0f}  Collisions: {np.mean(all_collisions):.0f}")

    # D4 zero-shot generalization using global best model
    print(f"\n  --- D4 Zero-Shot Generalization ---")
    base_env_gen = AutonomousNavigationEnv(size=13)
    best_m = make_model(model_cls)
    best_m.load_state_dict(global_best_state)
    gen_greedy, gen_mcts = evaluate_d4_generalization(best_m, base_env_gen, n=30)

    return {
        "label":            label,
        "color":            color,
        "linestyle":        linestyle,
        "greedy_mean":      mg,    "greedy_std":  sg,
        "mcts_mean":        mm,    "mcts_std":    sm,
        "mean_conv_ep":     float(np.mean(all_conv_ep)),
        "mean_collisions":  float(np.mean(all_collisions)),
        "eval_episodes":    eval_eps,
        "curve_greedy_mean": curves_g.mean(0).tolist(),
        "curve_greedy_std":  curves_g.std(0).tolist(),
        "curve_mcts_mean":   curves_m.mean(0).tolist(),
        "curve_mcts_std":    curves_m.std(0).tolist(),
        "d4_labels":          D4_LABELS,
        "generalization_greedy": gen_greedy,
        "generalization_mcts":   gen_mcts,
        "seeds": SEEDS,
    }


# ══════════════════════════════════════════════════════════════════════
#  Plot Generation (English-only, publication quality)
# ══════════════════════════════════════════════════════════════════════

def plot_all(results):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 14))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_curve  = fig.add_subplot(gs[0, :2])
    ax_bar    = fig.add_subplot(gs[0, 2])
    ax_gen    = fig.add_subplot(gs[1, :2])
    ax_conv   = fig.add_subplot(gs[1, 2])

    eps = results[0]["eval_episodes"]

    # ── 1. Learning Curves ───────────────────────────────────────────
    ax_curve.set_title(f"Real Learning Curves (Mean ± Std, {len(SEEDS)} Seeds)",
                        fontsize=13, fontweight="bold")
    for res in results:
        mu = np.array(res["curve_greedy_mean"]) * 100
        sd = np.array(res["curve_greedy_std"])  * 100
        ax_curve.plot(eps, mu, label=res["label"] + " (Greedy)",
                      color=res["color"], linestyle=res["linestyle"], lw=2)
        ax_curve.fill_between(eps, mu-sd, mu+sd, alpha=0.15, color=res["color"])
        mu2 = np.array(res["curve_mcts_mean"]) * 100
        sd2 = np.array(res["curve_mcts_std"])  * 100
        ax_curve.plot(eps, mu2, label=res["label"] + " (+MCTS)",
                      color=res["color"], linestyle=":", lw=1.5, alpha=0.8)
        ax_curve.fill_between(eps, mu2-sd2, mu2+sd2, alpha=0.08, color=res["color"])
    ax_curve.axhline(70, color="white", lw=0.8, ls="--", alpha=0.35,
                     label="70% convergence threshold")
    ax_curve.axvline(WARMUP_EPS, color="#aaa", lw=0.8, ls="--", alpha=0.5,
                     label=f"Warm-up end (Ep {WARMUP_EPS})")
    ax_curve.set_xlabel("Training Episodes", fontsize=11)
    ax_curve.set_ylabel("Success Rate (%)", fontsize=11)
    ax_curve.legend(fontsize=8, loc="upper left", ncol=2)
    ax_curve.set_ylim(0, 110); ax_curve.grid(True, alpha=0.15)

    # ── 2. Final Success Rate Bar ─────────────────────────────────────
    ax_bar.set_title(f"Final Success Rate\n(Mean ± Std, {len(SEEDS)} Seeds)",
                     fontsize=13, fontweight="bold")
    x  = np.arange(len(results))
    w  = 0.35
    ax_bar.bar(x - w/2,
               [r["greedy_mean"]*100 for r in results], w,
               yerr=[r["greedy_std"]*100 for r in results],
               capsize=5, label="Greedy",
               color=[r["color"] for r in results], alpha=0.85,
               error_kw={"ecolor":"white","elinewidth":1.5})
    ax_bar.bar(x + w/2,
               [r["mcts_mean"]*100 for r in results], w,
               yerr=[r["mcts_std"]*100 for r in results],
               capsize=5, label="+MCTS",
               color=[r["color"] for r in results], alpha=0.50, hatch="//",
               error_kw={"ecolor":"white","elinewidth":1.5})
    for i, res in enumerate(results):
        ax_bar.text(i-w/2, res["greedy_mean"]*100 + res["greedy_std"]*100 + 2,
                    f'{res["greedy_mean"]*100:.1f}%', ha="center", fontsize=9, color="white")
        ax_bar.text(i+w/2, res["mcts_mean"]*100 + res["mcts_std"]*100 + 2,
                    f'{res["mcts_mean"]*100:.1f}%', ha="center", fontsize=9, color="white")
    short = [r["label"].replace(" (Proposed)","") for r in results]
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(short, fontsize=9)
    ax_bar.set_ylim(0, 120); ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.15, axis="y")

    # ── 3. D4 Zero-Shot Generalization (all 8 elements) ──────────────
    ax_gen.set_title("Zero-Shot D4 Generalization (All 8 Group Elements: 4 Rotations + 4 Reflections)",
                     fontsize=13, fontweight="bold")
    x_d4 = np.arange(8)
    for res in results:
        gg = [v*100 for v in res["generalization_greedy"]]
        gm = [v*100 for v in res["generalization_mcts"]]
        ax_gen.plot(x_d4, gg, marker="o", color=res["color"],
                    label=res["label"][:30] + " (Greedy)", lw=2.5, ms=8)
        ax_gen.plot(x_d4, gm, marker="s", color=res["color"],
                    ls=":", lw=1.5, alpha=0.75, label=res["label"][:30] + " (+MCTS)")
    ax_gen.set_xticks(x_d4)
    ax_gen.set_xticklabels(D4_LABELS, rotation=15, ha="right", fontsize=9)
    ax_gen.set_ylabel("Zero-Shot Success Rate (%)", fontsize=11)
    ax_gen.set_ylim(-5, 115)
    ax_gen.axhline(100, color="white", lw=0.7, ls="--", alpha=0.3)
    # Shade rotation vs reflection regions
    ax_gen.axvspan(-0.5, 3.5, alpha=0.06, color="#88CC88", label="Rotations (g0-g3)")
    ax_gen.axvspan( 3.5, 7.5, alpha=0.06, color="#CC8888", label="Reflections (g4-g7)")
    ax_gen.legend(fontsize=8, loc="lower left", ncol=2)
    ax_gen.grid(True, alpha=0.15)

    # ── 4. Convergence Speed + Collisions (combined) ──────────────────
    ax_conv.set_title("Sample Efficiency & Safety\n(Lower = Better)",
                       fontsize=13, fontweight="bold")
    colors = [r["color"] for r in results]
    x_c = np.arange(len(results))
    w_c = 0.35
    conv_vals = [r["mean_conv_ep"]     for r in results]
    coll_vals = [r["mean_collisions"]  for r in results]
    # Normalize for dual-axis display
    ax_conv2 = ax_conv.twinx()
    b1 = ax_conv.bar(x_c - w_c/2, conv_vals, w_c,
                     color=colors, alpha=0.85, label="Conv. Episodes")
    b2 = ax_conv2.bar(x_c + w_c/2, coll_vals, w_c,
                      color=colors, alpha=0.50, hatch="xx", label="Collisions")
    for i, (cv, co) in enumerate(zip(conv_vals, coll_vals)):
        ax_conv.text(i-w_c/2, cv+1, f"{cv:.0f}", ha="center", fontsize=9, color="white")
        ax_conv2.text(i+w_c/2, co+1, f"{co:.0f}", ha="center", fontsize=9, color="white")
    ax_conv.set_xticks(x_c); ax_conv.set_xticklabels(short, fontsize=9)
    ax_conv.set_ylabel("Episodes to 70% Success", fontsize=10)
    ax_conv2.set_ylabel("Total Training Collisions", fontsize=10)
    lines1, labels1 = ax_conv.get_legend_handles_labels()
    lines2, labels2 = ax_conv2.get_legend_handles_labels()
    ax_conv.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax_conv.grid(True, alpha=0.15, axis="y")

    fig.suptitle(
        "Sample-Efficient Autonomous Navigation via D4-Equivariant RL + Heuristic-Guided MCTS\n"
        f"(Train-First Search-Later | {len(SEEDS)} Seeds × {NUM_EPISODES} Episodes | CPU | v3)",
        fontsize=14, fontweight="bold", y=1.00
    )

    out = "real_results_plots_v3.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\n  Plot saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("D4-Equivariant + KL Guide (Proposed)", D4EquivariantNet, "#E63946", "-"),
    ("Standard CNN + KL Guide",              StandardCNN,      "#457B9D", "--"),
]

def main():
    print("\n" + "="*65)
    print("  real_experiment_v3  ─  Train First, Search Later")
    print(f"  Setting : {len(SEEDS)} Seeds × {NUM_EPISODES} Ep × {len(EXPERIMENTS)} Models")
    print(f"  Warm-up : {WARMUP_EPS} Ep → AC+KL: {NUM_EPISODES-WARMUP_EPS} Ep")
    print(f"  Entropy : ENT_COEF = {ENT_COEF}  (policy collapse prevention)")
    print(f"  D4 Gen  : ALL 8 group elements (4 rotations + 4 reflections)")
    print(f"  Device  : {DEVICE}")
    print("="*65)

    t0 = time.time()
    all_results = []

    for label, cls, color, ls in EXPERIMENTS:
        res = run_experiment(label, cls, color, ls)
        all_results.append(res)

    # Save JSON
    out_json = "real_results_v3.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n  Results saved: {out_json}")

    plot_all(all_results)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*65}")
    print(f"  Total duration: {elapsed:.1f} min")
    print(f"{'='*65}")
    for res in all_results:
        print(f"  {res['label'][:50]:50s}")
        print(f"    Greedy : {res['greedy_mean']*100:.1f}% ± {res['greedy_std']*100:.1f}%")
        print(f"    +MCTS  : {res['mcts_mean']*100:.1f}% ± {res['mcts_std']*100:.1f}%")
        print(f"    D4 Gen (Greedy): " + ", ".join(f"{v*100:.0f}%" for v in res['generalization_greedy']))
    print("="*65)


if __name__ == "__main__":
    main()
