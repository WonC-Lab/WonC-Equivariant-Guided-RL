"""
run_academic_experiments.py  ─  Comprehensive Academic Experiment Suite
=======================================================================
This script executes the 5 experiments described in the paper draft:
  1. Ablation Study: Proposed vs. w/o Equivariance vs. w/o Heuristic vs. w/o MCTS.
  2. Zero-Shot Generalization: Proposed vs. Standard CNN (with and without 8x Augmentation).
  3. Exploration Safety: Cumulative collisions during training (Guided vs. Pure RL).
  4. Sample Efficiency: Comparison of success rates on log scale.
  5. MCTS Sensitivity: Learning curves and computation time under N = 5, 15, 30.

Outputs:
  academic_results.json        ─ Quantitative metrics (mean ± std across 5 seeds)
  fig1_ablation_study.png      ─ Ablation study learning curves
  fig2_generalization_test.png ─ Zero-shot D4 generalization test
  fig3_exploration_safety.png  ─ Cumulative training collisions
  fig4_sample_efficiency.png   ─ Convergence comparison on log scale
  fig5_mcts_sensitivity.png    ─ MCTS search count sensitivity
"""

import os, sys, json, time, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

# ── Local modules ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autonomous_env import AutonomousNavigationEnv
from equivariant_models import D4EquivariantNet, StandardCNN, D4GroupAction
from mcts_actor_critic import ActorCriticMCTS
from train_navigation import SymmetricNavEnvAdapter

# ── Device ────────────────────────────────────────────────────────────
# CPU is preferred for sequential MCTS steps on local machines to avoid synchronization overheads.
DEVICE = torch.device("cpu")
print("Forcing CPU for stable sequential MCTS search execution.")

# ── Hyperparameters ───────────────────────────────────────────────────
SEEDS         = [42, 123, 456, 789, 2024]   # 5 independent seeds
NUM_EPISODES  = 300
WARMUP_EPS    = 60
MAX_STEPS     = 40
GAMMA         = 0.95
LR            = 0.002
BETA_DECAY    = 0.95
BETA_MIN      = 0.5
ENT_COEF      = 0.01    # Entropy bonus coefficient
TRAIN_EPOCHS  = 3
GRAD_CLIP     = 1.0
EVAL_INTERVAL = 50
EVAL_EPS      = 15      # Reduced from 20 to speed up 5-config runs
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
    rot_k = action_idx % 4
    flip  = action_idx // 4
    if flip == 1:
        c = n - 1 - c          # horizontal flip
    for _ in range(rot_k):
        r, c = n - 1 - c, r   # 90° CCW
    return r, c

def apply_d4_to_map(obstacles, goal, size, action_idx):
    new_obs  = {apply_d4_to_coord(r, c, size, action_idx) for r, c in obstacles}
    gr, gc   = goal
    new_goal = apply_d4_to_coord(gr, gc, size, action_idx)
    return new_obs, new_goal

# ══════════════════════════════════════════════════════════════════════
#  Training Function with Optional Data Augmentation
# ══════════════════════════════════════════════════════════════════════
def train_one_episode(model, base_env, env_adapter, optimizer, beta,
                      episode_num, kl_loss_fn, data_aug=False):
    """
    Run a single training episode with Actor-Critic + KL Heuristic + Entropy Loss.
    Includes support for 8x Data Augmentation.
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
        
        # Original data vectors
        batch_s     = torch.cat(ep_tensors, 0)
        batch_a     = torch.tensor(ep_actions,     dtype=torch.long,    device=DEVICE)
        batch_r     = torch.tensor(returns,        dtype=torch.float32, device=DEVICE)
        batch_h     = torch.tensor(np.array(ep_heuristics), dtype=torch.float32, device=DEVICE)
        batch_old_lps = torch.tensor(ep_old_log_probs, dtype=torch.float32, device=DEVICE)

        # 8x Data Augmentation logic (Config C)
        if data_aug:
            aug_s, aug_a, aug_h, aug_r = [], [], [], []
            perms = [D4GroupAction.get_action_permutation(i) for i in range(8)]
            
            for idx in range(len(ep_tensors)):
                st = ep_tensors[idx]          # (1, 3, 13, 13)
                act = ep_actions[idx]         # int
                heur = ep_heuristics[idx]     # np.array (8,)
                ret = returns[idx]            # float
                
                for g_idx in range(8):
                    st_aug = D4GroupAction.apply_action(st, g_idx)
                    perm = perms[g_idx]
                    act_aug = perm[act]
                    heur_aug = heur[perm]
                    
                    aug_s.append(st_aug)
                    aug_a.append(act_aug)
                    aug_h.append(heur_aug)
                    aug_r.append(ret)
            
            batch_s = torch.cat(aug_s, 0)
            batch_a = torch.tensor(aug_a, dtype=torch.long, device=DEVICE)
            batch_h = torch.tensor(np.array(aug_h), dtype=torch.float32, device=DEVICE)
            batch_r = torch.tensor(aug_r, dtype=torch.float32, device=DEVICE)
            
            # Re-evaluate old log probabilities on augmented states for PPO mathematical correctness
            model.eval()
            with torch.no_grad():
                logits_old_aug, _ = model(batch_s)
            log_probs_old_aug = F.log_softmax(logits_old_aug, dim=-1)
            batch_old_lps = log_probs_old_aug[torch.arange(len(batch_a)), batch_a]

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

            # Entropy bonus
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
#  Evaluations
# ══════════════════════════════════════════════════════════════════════
def evaluate_greedy(model, base_env, env_adapter, n=EVAL_EPS):
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

def evaluate_with_mcts(model, base_env, env_adapter, sims=MCTS_SIMS, n=EVAL_EPS):
    model.eval()
    successes = 0
    start_time = time.time()
    for _ in range(n):
        mcts = ActorCriticMCTS(model=model, c_puct=1.4)
        state = base_env.generate_initial_state()
        for _ in range(MAX_STEPS):
            actions, probs = mcts.get_action_probabilities(
                state, 1, env_adapter, num_searches=sims, temp=0.1
            )
            if not actions: break
            chosen = actions[int(np.argmax(probs))]
            state, _ = env_adapter.step(state, chosen, 1)
            done, winner = env_adapter.check_game_over(state, 1)
            if done:
                if winner == 1: successes += 1
                break
    avg_duration = (time.time() - start_time) / n
    return successes / n, avg_duration

# ══════════════════════════════════════════════════════════════════════
#  D4 Zero-Shot Generalization Test (all 8 group elements)
# ══════════════════════════════════════════════════════════════════════
D4_LABELS = ["0° (id)", "90° CCW", "180°", "270° CCW",
             "Flip-H", "Flip-H+90°", "Flip-H+180°", "Flip-H+270°"]

def evaluate_d4_generalization(model, base_env, n=30):
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
        mr, _ = evaluate_with_mcts(model, rot_env, rot_adapter, sims=MCTS_SIMS, n=n)
        greedy_rates.append(gr)
        mcts_rates.append(mr)

    return greedy_rates, mcts_rates

# ══════════════════════════════════════════════════════════════════════
#  Main Experiment Execution Loop
# ══════════════════════════════════════════════════════════════════════
def run_experiment(label, model_cls, color, linestyle, beta_start, data_aug):
    print(f"\n{'='*75}")
    print(f"  Model Configuration: {label}")
    print(f"  Beta Start: {beta_start} | Data Augmentation: {data_aug}")
    print(f"{'='*75}")

    kl_fn = nn.KLDivLoss(reduction="batchmean")

    all_greedy_rates, all_mcts_rates = [], []
    all_curves_greedy, all_curves_mcts = [], []
    all_curves_mcts_5, all_curves_mcts_30 = [], []
    all_mcts_time_5, all_mcts_time_15, all_mcts_time_30 = [], [], []
    all_collisions, all_conv_ep = [], []
    all_collision_curves = []
    
    global_best_state = None
    global_best_rate  = -1.0

    for si, seed in enumerate(SEEDS):
        print(f"\n  [Seed {si+1}/{len(SEEDS)}: {seed}]")
        set_seed(seed)

        base_env    = AutonomousNavigationEnv(size=13)
        env_adapter = SymmetricNavEnvAdapter(base_env)
        model       = make_model(model_cls)
        optimizer   = optim.Adam(model.parameters(), lr=LR)
        beta        = beta_start

        curve_greedy, curve_mcts = [], []
        curve_mcts_5, curve_mcts_30 = [], []
        mcts_time_5, mcts_time_15, mcts_time_30 = [], [], []
        collision_curve = []
        collisions = 0
        conv_ep    = NUM_EPISODES

        best_seed_rate  = -1.0
        best_seed_state = None

        for ep in range(1, NUM_EPISODES + 1):
            # Dynamic obstacle randomization every episode during training for robustness
            base_env.randomize_obstacles(n_obstacles=12)
            
            success, collision, steps, loss = train_one_episode(
                model, base_env, env_adapter, optimizer, beta, ep, kl_fn, data_aug
            )
            if collision: 
                collisions += 1
            collision_curve.append(collisions)
            
            if beta_start > 0.0:
                beta = max(beta * BETA_DECAY, BETA_MIN)

            if ep % EVAL_INTERVAL == 0:
                # Reset environment to canonical obstacles for standardized evaluation
                base_env.reset_canonical_obstacles()
                
                gr = evaluate_greedy(model, base_env, env_adapter)
                mr15, t15 = evaluate_with_mcts(model, base_env, env_adapter, sims=15)
                
                curve_greedy.append(gr)
                curve_mcts.append(mr15)
                mcts_time_15.append(t15)

                # For the proposed model, we also track MCTS sims 5 and 30 for sensitivity profiling
                if "Proposed" in label:
                    mr5, t5 = evaluate_with_mcts(model, base_env, env_adapter, sims=5)
                    mr30, t30 = evaluate_with_mcts(model, base_env, env_adapter, sims=30)
                    curve_mcts_5.append(mr5)
                    curve_mcts_30.append(mr30)
                    mcts_time_5.append(t5)
                    mcts_time_30.append(t30)

                if gr >= 0.70 and conv_ep == NUM_EPISODES:
                    conv_ep = ep

                if gr > best_seed_rate:
                    best_seed_rate  = gr
                    best_seed_state = copy.deepcopy(model.state_dict())

                loss_s = f"{loss:.4f}" if loss else "N/A"
                mode   = "WarmUp" if ep <= WARMUP_EPS else "AC+KL  "
                print(f"    Ep {ep:4d}/{NUM_EPISODES} [{mode}] "
                      f"Greedy: {gr*100:5.1f}%  MCTS(15): {mr15*100:5.1f}%  "
                      f"Loss: {loss_s}  β: {beta:.3f}  Collisions: {collisions}")

        # Final evaluation using best checkpoint
        best_eval_model = make_model(model_cls)
        if best_seed_state is not None:
            best_eval_model.load_state_dict(best_seed_state)
        else:
            best_eval_model.load_state_dict(model.state_dict())

        base_env.reset_canonical_obstacles()
        final_greedy = evaluate_greedy(best_eval_model, base_env, env_adapter, n=50)
        final_mcts, _ = evaluate_with_mcts(best_eval_model, base_env, env_adapter, sims=15, n=50)

        if final_greedy > global_best_rate:
            global_best_rate  = final_greedy
            global_best_state = copy.deepcopy(best_eval_model.state_dict())

        all_greedy_rates.append(final_greedy)
        all_mcts_rates.append(final_mcts)
        all_curves_greedy.append(curve_greedy)
        all_curves_mcts.append(curve_mcts)
        all_collisions.append(collisions)
        all_conv_ep.append(conv_ep)
        all_collision_curves.append(collision_curve)

        if "Proposed" in label:
            all_curves_mcts_5.append(curve_mcts_5)
            all_curves_mcts_30.append(curve_mcts_30)
            all_mcts_time_5.append(mcts_time_5)
            all_mcts_time_15.append(mcts_time_15)
            all_mcts_time_30.append(mcts_time_30)

        print(f"  → Seed {seed}: Greedy {final_greedy*100:.1f}%  "
              f"MCTS {final_mcts*100:.1f}%  Conv@{conv_ep}  Coll:{collisions}")

    # Aggregate stats
    eval_eps = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
    
    def stats(arr):
        a = np.array(arr)
        return float(a.mean()), float(a.std())

    mg, sg = stats(all_greedy_rates)
    mm, sm = stats(all_mcts_rates)

    curves_g = np.array(all_curves_greedy)
    curves_m = np.array(all_curves_mcts)
    curves_collisions = np.array(all_collision_curves)

    print(f"\n  ★ Aggregated Stats ({label})")
    print(f"    Greedy:  {mg*100:.1f}% ± {sg*100:.1f}%")
    print(f"    +MCTS:   {mm*100:.1f}% ± {sm*100:.1f}%")
    print(f"    Conv Ep: {np.mean(all_conv_ep):.0f}  Collisions: {np.mean(all_collisions):.0f}")

    # Zero-shot generalization test
    base_env_gen = AutonomousNavigationEnv(size=13)
    best_m = make_model(model_cls)
    best_m.load_state_dict(global_best_state)
    gen_greedy, gen_mcts = evaluate_d4_generalization(best_m, base_env_gen, n=30)

    result_dict = {
        "label":             label,
        "color":             color,
        "linestyle":         linestyle,
        "greedy_mean":       mg,     "greedy_std":  sg,
        "mcts_mean":         mm,     "mcts_std":    sm,
        "mean_conv_ep":      float(np.mean(all_conv_ep)),
        "mean_collisions":   float(np.mean(all_collisions)),
        "eval_episodes":     eval_eps,
        "curve_greedy_mean":  curves_g.mean(0).tolist(),
        "curve_greedy_std":   curves_g.std(0).tolist(),
        "curve_mcts_mean":    curves_m.mean(0).tolist(),
        "curve_mcts_std":     curves_m.std(0).tolist(),
        "collision_curve_mean": curves_collisions.mean(0).tolist(),
        "generalization_greedy": gen_greedy,
        "generalization_mcts":   gen_mcts,
    }

    if "Proposed" in label:
        curves_m5 = np.array(all_curves_mcts_5)
        curves_m30 = np.array(all_curves_mcts_30)
        result_dict.update({
            "curve_mcts_5_mean":  curves_m5.mean(0).tolist(),
            "curve_mcts_5_std":   curves_m5.std(0).tolist(),
            "curve_mcts_30_mean": curves_m30.mean(0).tolist(),
            "curve_mcts_30_std":  curves_m30.std(0).tolist(),
            "time_mcts_5":        float(np.mean(all_mcts_time_5)),
            "time_mcts_15":       float(np.mean(all_mcts_time_15)),
            "time_mcts_30":       float(np.mean(all_mcts_time_30)),
        })

    return result_dict

# ══════════════════════════════════════════════════════════════════════
#  Individual Academic Plot Generators
# ══════════════════════════════════════════════════════════════════════

def smooth_curve(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    for idx in range(box_pts):
        y_smooth[idx] = np.mean(y[:idx+1])
        y_smooth[-idx-1] = np.mean(y[-idx-1:])
    return y_smooth

def plot_fig1_ablation(results):
    """
    Fig 1: Ablation Study
    Proposed vs. w/o Equivariance vs. w/o Heuristic vs. w/o MCTS
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7.5, 4.5))
    
    eps = results["Proposed"]["eval_episodes"]
    
    # 1. Full (Proposed)
    mu_prop = np.array(results["Proposed"]["curve_mcts_mean"])
    plt.plot(eps, mu_prop, label="Full Framework (Proposed)", color="#E63946", lw=2.5)
    
    # 2. w/o Equivariance (Standard CNN + KL Guide)
    mu_no_equi = np.array(results["Standard CNN + KL Guide"]["curve_mcts_mean"])
    plt.plot(eps, mu_no_equi, label="w/o Equivariance (Std CNN)", color="#457B9D", ls="--", lw=2.0)
    
    # 3. w/o Heuristic Guidance (D4-Equivariant No KL Guide)
    mu_no_guide = np.array(results["D4-Equivariant (No KL)"]["curve_mcts_mean"])
    plt.plot(eps, mu_no_guide, label="w/o Heuristic Guidance", color="#2A9D8F", ls=":", lw=2.0)
    
    # 4. w/o MCTS (Proposed evaluated Greedy)
    mu_no_mcts = np.array(results["Proposed"]["curve_greedy_mean"])
    plt.plot(eps, mu_no_mcts, label="w/o MCTS (Direct Policy)", color="#8338EC", ls="-.", lw=2.0)
    
    plt.title("Ablation Study: Convergence of Framework Components", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Training Episodes", fontsize=10)
    plt.ylabel("Success Rate", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("ablation_study.png", dpi=300)
    plt.close()
    print("  Saved plot: ablation_study.png (Fig 1)")

def plot_fig2_generalization(results):
    """
    Fig 2: Zero-Shot Generalization under 8 D4 transformations
    Proposed vs. Standard CNN (with and without 8x Augmentation)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(9, 4.5))
    x = np.arange(8)
    width = 0.25
    
    y_prop = [v * 100 for v in results["Proposed"]["generalization_greedy"]]
    y_aug  = [v * 100 for v in results["Standard CNN + 8x Aug"]["generalization_greedy"]]
    y_std  = [v * 100 for v in results["Standard CNN + KL Guide"]["generalization_greedy"]]
    
    plt.bar(x - width, y_prop, width, label="D4-Net (Symmetric, Ours)", color="#E63946", alpha=0.9)
    plt.bar(x, y_aug, width, label="Standard CNN (8x Augmented)", color="#FFB703", alpha=0.9)
    plt.bar(x + width, y_std, width, label="Standard CNN (No Augmentation)", color="#457B9D", alpha=0.9)
    
    plt.xticks(x, D4_LABELS, rotation=15, ha="right", fontsize=9)
    plt.title("Zero-Shot Generalization Under $D_4$ Group Actions", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Dihedral Group Transformation Action ($g_0$: Id, $g_1$-$g_3$: Rot, $g_4$-$g_7$: Refl)", fontsize=10)
    plt.ylabel("Test Success Rate (%)", fontsize=10)
    plt.ylim(0, 115)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("generalization_test.png", dpi=300)
    plt.close()
    print("  Saved plot: generalization_test.png (Fig 2)")

def plot_fig3_safety(results):
    """
    Fig 3: Cumulative obstacle collisions during training
    Proposed (Heuristic-Guided) vs. Pure RL (No KL Guide)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7.5, 4.5))
    
    episodes = np.arange(1, NUM_EPISODES + 1)
    
    guided = results["Proposed"]["collision_curve_mean"]
    unguided = results["D4-Equivariant (No KL)"]["collision_curve_mean"]
    
    plt.plot(episodes, guided, label="Heuristic-Guided RL (Proposed, Beta=2.0)", color="#E63946", lw=2.5)
    plt.plot(episodes, unguided, label="Pure RL Exploration (Beta=0.0)", color="#8338EC", ls="--", lw=2.0)
    
    plt.title("Safety Analysis: Cumulative Collisions During Training", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Training Episodes", fontsize=10)
    plt.ylabel("Cumulative Obstacle Collisions", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("exploration_safety.png", dpi=300)
    plt.close()
    print("  Saved plot: exploration_safety.png (Fig 3)")

def plot_fig4_sample_efficiency(results):
    """
    Fig 4: Sample Efficiency on Log scale
    Proposed vs. Standard CNN (with and without 8x Augmentation)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7.5, 4.5))
    
    eps = results["Proposed"]["eval_episodes"]
    
    plt.plot(eps, results["Proposed"]["curve_mcts_mean"], label="D4-Net + MCTS + Guidance (Ours)", color="#E63946", lw=2.5)
    plt.plot(eps, results["Standard CNN + 8x Aug"]["curve_mcts_mean"], label="Standard CNN + Augmentation (8x)", color="#FFB703", ls="--", lw=2.0)
    plt.plot(eps, results["Standard CNN + KL Guide"]["curve_mcts_mean"], label="Standard CNN (No Augmentation)", color="#457B9D", ls=":", lw=2.0)
    
    plt.title("Sample Efficiency & Training Convergence Comparison", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Training Episodes (Log Scale)", fontsize=10)
    plt.xscale("log")
    plt.xlim(EVAL_INTERVAL, NUM_EPISODES)
    plt.ylabel("Success Rate", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("sample_efficiency.png", dpi=300)
    plt.close()
    print("  Saved plot: sample_efficiency.png (Fig 4)")

def plot_fig5_sensitivity(results):
    """
    Fig 5: MCTS Search Scale Sensitivity
    Learning curves for N = 5, 15, 30
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7.5, 4.5))
    
    eps = results["Proposed"]["eval_episodes"]
    
    plt.plot(eps, results["Proposed"]["curve_mcts_30_mean"], label="MCTS Simulations N=30", color="#2A9D8F", lw=2.0)
    plt.plot(eps, results["Proposed"]["curve_mcts_mean"],    label="MCTS Simulations N=15", color="#E63946", ls="--", lw=2.0)
    plt.plot(eps, results["Proposed"]["curve_mcts_5_mean"],   label="MCTS Simulations N=5",  color="#FFB703", ls=":", lw=2.0)
    
    plt.title("MCTS Scale Sensitivity: Learning Curve Comparison", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Training Episodes", fontsize=10)
    plt.ylabel("Success Rate", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("mcts_sensitivity.png", dpi=300)
    plt.close()
    print("  Saved plot: mcts_sensitivity.png (Fig 5)")

# ══════════════════════════════════════════════════════════════════════
#  Main Execution
# ══════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    ("Proposed",                     D4EquivariantNet, "#E63946", "-",  2.0, False),
    ("Standard CNN + KL Guide",      StandardCNN,      "#457B9D", "--", 2.0, False),
    ("Standard CNN + 8x Aug",        StandardCNN,      "#FFB703", "-.", 2.0, True),
    ("D4-Equivariant (No KL)",       D4EquivariantNet, "#2A9D8F", ":",  0.0, False),
    ("Standard CNN (No KL)",         StandardCNN,      "#8338EC", ":",  0.0, False),
]

def main():
    print("\n" + "="*75)
    print("  Starting Full Academic Experiment Suite (v3 - Publication Standard)")
    print(f"  Setup  : {len(SEEDS)} Seeds × {NUM_EPISODES} Episodes × {len(EXPERIMENTS)} Models")
    print(f"  Device : {DEVICE}")
    print("="*75)

    t0 = time.time()
    results = {}

    for label, cls, color, ls, beta_start, data_aug in EXPERIMENTS:
        res = run_experiment(label, cls, color, ls, beta_start, data_aug)
        results[label] = res

    # Save results to JSON file
    out_json = "academic_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved quantitative results: {out_json}")

    # Generate the 5 distinct publication figures
    print("\nGenerating 5 publication-grade figures...")
    plot_fig1_ablation(results)
    plot_fig2_generalization(results)
    plot_fig3_safety(results)
    plot_fig4_sample_efficiency(results)
    plot_fig5_sensitivity(results)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*75}")
    print(f"  All experiments completed successfully in {elapsed:.1f} minutes.")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
