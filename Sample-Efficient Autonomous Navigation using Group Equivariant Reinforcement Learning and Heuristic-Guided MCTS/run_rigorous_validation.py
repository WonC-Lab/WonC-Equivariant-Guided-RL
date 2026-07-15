"""
run_rigorous_validation.py  ─  Rigorous Validation Suite for Academic Publication
===================================================================================
This script executes a rigorous validation pipeline:
  1. Train models on dynamically randomized obstacles (to ensure they don't memorize a single path).
  2. Intermediate validation: Evaluate on 20 unseen random layouts (RNG seed 5555).
  3. Final evaluation: Evaluate best checkpoints on 50 unseen random layouts (RNG seed 9999).
  4. Obstacle Density Robustness: Evaluate on 30 maps of low (8), medium (12), and high (16) obstacle density (RNG seed 8888).
  5. Zero-Shot Generalization on Unseen Layouts: Evaluate on 30 unseen random layouts (RNG seed 7777) rotated/reflected under all 8 D4 transformations.

Outputs:
  academic_results_rigorous.json      ─ Quantitative results
  ablation_study_unseen.png           ─ Learning curves on unseen maps
  generalization_test_unseen.png      ─ Zero-shot D4 generalization on unseen layouts
  density_robustness.png              ─ Success rate vs. obstacle density
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, json, time, math, random, copy, argparse
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
DEVICE = torch.device("cpu")
print("Forcing CPU for stable sequential MCTS search execution.")

# ── CLI Arguments ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Rigorous Validation Suite")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run (1 to 10)")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[13], help="Grid dimension sizes (e.g. 13 21)")
    parser.add_argument("--quick", action="store_true", help="Run a quick verification test with minimal workloads")
    return parser.parse_args()

args = parse_args()

# ── Seed configurations (up to 10 seeds) ──────────────────────────────
ALL_SEEDS = [42, 123, 456, 789, 2024, 999, 888, 777, 111, 222]

if args.quick:
    SEEDS         = ALL_SEEDS[:2]
    NUM_EPISODES  = 5
    WARMUP_EPS    = 2
    MCTS_SIMS     = 5
    EVAL_INTERVAL = 2
else:
    num_seeds = min(max(args.seeds, 1), 10)
    SEEDS         = ALL_SEEDS[:num_seeds]
    NUM_EPISODES  = args.episodes
    WARMUP_EPS    = int(NUM_EPISODES * 0.2)
    MCTS_SIMS     = 15
    EVAL_INTERVAL = 50

GRID_SIZE     = 13
MAX_STEPS     = 39  # scaled in main loop dynamically
GAMMA         = 0.95
LR            = 0.002
BETA_START    = 2.0
BETA_DECAY    = 0.95
BETA_MIN      = 0.5
ENT_COEF      = 0.01    # Entropy bonus coefficient
TRAIN_EPOCHS  = 3
GRAD_CLIP     = 1.0

NUM_LAYERS    = 4
NUM_FILTERS   = 16

# ── Utilities ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_model(model_cls, size=None):
    if size is None:
        size = GRID_SIZE
    m = model_cls(board_size=size, in_channels=3,
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

# ── Generate Validation/Test datasets with fixed seeds ─────────────────
def generate_unseen_maps(num_maps, n_obstacles, seed, size=None):
    if size is None:
        size = GRID_SIZE
    rng = np.random.RandomState(seed)
    maps = []
    temp_env = AutonomousNavigationEnv(size=size)
    for _ in range(num_maps):
        temp_env.randomize_obstacles(n_obstacles=n_obstacles, rng=rng)
        maps.append((copy.deepcopy(temp_env.obstacles), temp_env.start, temp_env.goal))
    return maps

VAL_MAPS_20 = generate_unseen_maps(20, int(GRID_SIZE * 0.9), 5555, GRID_SIZE)  # 20 unseen maps for validation curve
TEST_MAPS_50 = generate_unseen_maps(50, int(GRID_SIZE * 0.9), 9999, GRID_SIZE) # 50 unseen maps for final test

LOW_DENS  = int(GRID_SIZE * 0.6)
MED_DENS  = int(GRID_SIZE * 0.9)
HIGH_DENS = int(GRID_SIZE * 1.2)

DENSITY_MAPS = {
    LOW_DENS:  generate_unseen_maps(30, LOW_DENS, 8888, GRID_SIZE),        # Low density
    MED_DENS:  generate_unseen_maps(30, MED_DENS, 8888, GRID_SIZE),        # Medium density
    HIGH_DENS: generate_unseen_maps(30, HIGH_DENS, 8888, GRID_SIZE),       # High density
}
GEN_MAPS_30 = generate_unseen_maps(30, int(GRID_SIZE * 0.9), 7777, GRID_SIZE)  # 30 maps for symmetry generalization tests

# ══════════════════════════════════════════════════════════════════════
#  Training Function (Dynamic Obstacles)
# ══════════════════════════════════════════════════════════════════════
def train_one_episode(model, base_env, env_adapter, optimizer, beta,
                      episode_num, kl_loss_fn, data_aug=False):
    """
    Run a single training episode on a randomly generated obstacle layout.
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

        if data_aug:
            aug_s, aug_a, aug_h, aug_r = [], [], [], []
            perms = [D4GroupAction.get_action_permutation(i) for i in range(8)]
            for idx in range(len(ep_tensors)):
                st = ep_tensors[idx]
                act = ep_actions[idx]
                heur = ep_heuristics[idx]
                ret = returns[idx]
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

            model.eval()
            with torch.no_grad():
                logits_old_aug, _ = model(batch_s)
            log_probs_old_aug = F.log_softmax(logits_old_aug, dim=-1)
            batch_old_lps = log_probs_old_aug[torch.arange(len(batch_a)), batch_a]

        model.train()
        total_loss = 0.0
        for _ in range(TRAIN_EPOCHS):
            optimizer.zero_grad()
            logits, values = model(batch_s)
            log_probs = F.log_softmax(logits, dim=-1)
            sel_lp    = log_probs[torch.arange(len(batch_a)), batch_a]

            with torch.no_grad():
                adv = batch_r - values.squeeze(-1)

            ratio = torch.exp(sel_lp - batch_old_lps)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
            pg_loss = -torch.min(surr1, surr2).mean()

            kl = kl_loss_fn(log_probs, batch_h)
            val_loss = F.mse_loss(values.squeeze(-1), batch_r)

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
#  Evaluation on Unseen Maps (Rigorous)
# ══════════════════════════════════════════════════════════════════════
def evaluate_on_unseen(model, unseen_maps, use_mcts=False, sims=MCTS_SIMS, size=None):
    if size is None:
        size = GRID_SIZE
    model.eval()
    successes = 0
    env = AutonomousNavigationEnv(size=size)
    adapter = SymmetricNavEnvAdapter(env)

    for obs, start, goal in unseen_maps:
        env.obstacles = set(obs)
        env.start = start
        env.goal = goal
        state = env.generate_initial_state()

        if use_mcts:
            mcts = ActorCriticMCTS(model=model, c_puct=1.4)

        game_over = False
        steps = 0
        while not game_over and steps < MAX_STEPS:
            if use_mcts:
                actions, probs = mcts.get_action_probabilities(
                    state, 1, adapter, num_searches=sims, temp=0.1
                )
                if not actions: break
                chosen = actions[int(np.argmax(probs))]
            else:
                valid = env.get_valid_actions(state)
                if not valid: break
                st = adapter.state_to_tensor(state).to(DEVICE)
                with torch.no_grad():
                    logits, _ = model(st)
                lnp = logits.squeeze(0).cpu().numpy()
                mask = np.full(8, -1e9)
                for di in valid:
                    mask[di] = lnp[di]
                chosen = int(np.argmax(mask))

            state, _ = adapter.step(state, chosen, 1)
            done, winner = env.check_game_over(state, 1)
            if done:
                if winner == 1:
                    successes += 1
                break
            steps += 1

    return successes / len(unseen_maps)

# ══════════════════════════════════════════════════════════════════════
#  Zero-Shot Generalization Test on Unseen Maps
# ══════════════════════════════════════════════════════════════════════
def evaluate_d4_generalization_unseen(model, unseen_maps, size=None):
    if size is None:
        size = GRID_SIZE
    greedy_rates = []
    for g_idx in range(8):
        transformed_maps = []
        for obs, start, goal in unseen_maps:
            new_obs, new_goal = apply_d4_to_map(obs, goal, size, g_idx)
            new_start = apply_d4_to_coord(start[0], start[1], size, g_idx)
            transformed_maps.append((new_obs, new_start, new_goal))
        
        gr = evaluate_on_unseen(model, transformed_maps, use_mcts=False, size=size)
        greedy_rates.append(gr)
    return greedy_rates

# ══════════════════════════════════════════════════════════════════════
#  Experiment Running Loop
# ══════════════════════════════════════════════════════════════════════
def run_experiment(label, model_cls, color, linestyle, beta_start, data_aug):
    print(f"\n{'='*75}")
    print(f"  Rigorous Experiment Setup: {label}")
    print(f"  Beta Start: {beta_start} | Data Augmentation: {data_aug}")
    print(f"{'='*75}")

    kl_fn = nn.KLDivLoss(reduction="batchmean")

    all_test_greedy, all_test_mcts = [], []
    all_val_curves_greedy, all_val_curves_mcts = [], []
    all_collisions, all_conv_ep = [], []
    all_density_results = {LOW_DENS: [], MED_DENS: [], HIGH_DENS: []}

    global_best_state = None
    global_best_rate  = -1.0

    for si, seed in enumerate(SEEDS):
        print(f"\n  [Seed {si+1}/{len(SEEDS)}: {seed}]")
        set_seed(seed)

        base_env    = AutonomousNavigationEnv(size=GRID_SIZE)
        env_adapter = SymmetricNavEnvAdapter(base_env)
        model       = make_model(model_cls, GRID_SIZE)
        optimizer   = optim.Adam(model.parameters(), lr=LR)
        beta        = beta_start

        val_curve_g, val_curve_m = [], []
        collisions = 0
        conv_ep    = NUM_EPISODES

        best_seed_rate  = -1.0
        best_seed_state = None

        for ep in range(1, NUM_EPISODES + 1):
            # Dynamic obstacle randomization scaled with board size
            base_env.randomize_obstacles(n_obstacles=int(GRID_SIZE * 0.9))

            success, collision, steps, loss = train_one_episode(
                model, base_env, env_adapter, optimizer, beta, ep, kl_fn, data_aug
            )
            if collision:
                collisions += 1

            if beta_start > 0.0:
                beta = max(beta * BETA_DECAY, BETA_MIN)

            # Intermediate validation on 20 unseen random layouts
            if ep % EVAL_INTERVAL == 0:
                gr = evaluate_on_unseen(model, VAL_MAPS_20, use_mcts=False, size=GRID_SIZE)
                mr = evaluate_on_unseen(model, VAL_MAPS_20, use_mcts=True, size=GRID_SIZE)
                val_curve_g.append(gr)
                val_curve_m.append(mr)

                if gr >= 0.70 and conv_ep == NUM_EPISODES:
                    conv_ep = ep

                if gr > best_seed_rate:
                    best_seed_rate  = gr
                    best_seed_state = copy.deepcopy(model.state_dict())

                loss_s = f"{loss:.4f}" if loss else "N/A"
                print(f"    Ep {ep:4d}/{NUM_EPISODES} [Train] "
                      f"Val Greedy: {gr*100:5.1f}%  Val MCTS: {mr*100:5.1f}%  "
                      f"Loss: {loss_s}  Collisions: {collisions}")

        # Final evaluation of seed best checkpoint on 50 unseen random maps
        best_eval_model = make_model(model_cls, GRID_SIZE)
        if best_seed_state is not None:
            best_eval_model.load_state_dict(best_seed_state)
        else:
            best_eval_model.load_state_dict(model.state_dict())

        test_greedy = evaluate_on_unseen(best_eval_model, TEST_MAPS_50, use_mcts=False, size=GRID_SIZE)
        test_mcts   = evaluate_on_unseen(best_eval_model, TEST_MAPS_50, use_mcts=True, size=GRID_SIZE)

        if test_greedy > global_best_rate:
            global_best_rate  = test_greedy
            global_best_state = copy.deepcopy(best_eval_model.state_dict())

        all_test_greedy.append(test_greedy)
        all_test_mcts.append(test_mcts)
        all_val_curves_greedy.append(val_curve_g)
        all_val_curves_mcts.append(val_curve_m)
        all_collisions.append(collisions)
        all_conv_ep.append(conv_ep)

        # Obstacle Density Robustness test
        for dens in [LOW_DENS, MED_DENS, HIGH_DENS]:
            dens_rate = evaluate_on_unseen(best_eval_model, DENSITY_MAPS[dens], use_mcts=False, size=GRID_SIZE)
            all_density_results[dens].append(dens_rate)

        print(f"  → Seed {seed}: Test Greedy {test_greedy*100:.1f}%  "
              f"Test MCTS {test_mcts*100:.1f}%  Conv@{conv_ep}  Coll:{collisions}")

    # Aggregate statistics
    eval_eps = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
    
    def stats(arr):
        a = np.array(arr)
        return float(a.mean()), float(a.std())

    mg, sg = stats(all_test_greedy)
    mm, sm = stats(all_test_mcts)

    val_g_mu = np.mean(all_val_curves_greedy, axis=0).tolist()
    val_g_sd = np.std(all_val_curves_greedy, axis=0).tolist()
    val_m_mu = np.mean(all_val_curves_mcts, axis=0).tolist()
    val_m_sd = np.std(all_val_curves_mcts, axis=0).tolist()

    # Density stats
    dens_stats = {}
    for dens in [LOW_DENS, MED_DENS, HIGH_DENS]:
        d_mu, d_sd = stats(all_density_results[dens])
        dens_stats[dens] = {"mean": d_mu, "std": d_sd}

    # Zero-shot generalization on unseen layouts using global best checkpoint
    best_m = make_model(model_cls, GRID_SIZE)
    best_m.load_state_dict(global_best_state)
    gen_rates = evaluate_d4_generalization_unseen(best_m, GEN_MAPS_30)

    print(f"\n  ★ Aggregated Stats ({label}) on Unseen Maps")
    print(f"    Greedy Success: {mg*100:.1f}% ± {sg*100:.1f}%")
    print(f"    MCTS Success:   {mm*100:.1f}% ± {sm*100:.1f}%")
    print(f"    Conv Episode:   {np.mean(all_conv_ep):.0f}  Collisions: {np.mean(all_collisions):.0f}")

    return {
        "label":               label,
        "color":               color,
        "linestyle":           linestyle,
        "greedy_mean":         mg,       "greedy_std":  sg,
        "mcts_mean":           mm,       "mcts_std":    sm,
        "mean_conv_ep":        float(np.mean(all_conv_ep)),
        "mean_collisions":     float(np.mean(all_collisions)),
        "eval_episodes":       eval_eps,
        "val_greedy_mean":     val_g_mu, "val_greedy_std": val_g_sd,
        "val_mcts_mean":       val_m_mu, "val_mcts_std":   val_m_sd,
        "density_results":     dens_stats,
        "generalization_greedy": gen_rates,
    }

# ══════════════════════════════════════════════════════════════════════
#  Plotting Functions
# ══════════════════════════════════════════════════════════════════════
def plot_learning_curves_unseen(results, size=13):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7.5, 4.5))
    eps = results["Proposed"]["eval_episodes"]

    for label, res in results.items():
        mu = np.array(res["val_mcts_mean"]) * 100
        sd = np.array(res["val_mcts_std"]) * 100
        plt.plot(eps, mu, label=res["label"] + " (+MCTS)", color=res["color"], ls=res["linestyle"], lw=2)
        plt.fill_between(eps, mu-sd, mu+sd, alpha=0.12, color=res["color"])

    plt.title(f"Ablation Study: Success Rate on Unseen Obstacle Maps ({size}x{size})", fontsize=12, fontweight="bold")
    plt.xlabel("Training Episodes", fontsize=10)
    plt.ylabel("Validation Success Rate (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"ablation_study_unseen_{size}.png", dpi=300)
    plt.close()
    print(f"  Saved plot: ablation_study_unseen_{size}.png")

def plot_generalization_unseen(results, size=13):
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
    plt.title(f"Zero-Shot Generalization on Unseen Random Maps under $D_4$ Symmetries ({size}x{size})", fontsize=12, fontweight="bold")
    plt.xlabel("Dihedral Group Transformation Action ($g_0$: Id, $g_1$-$g_3$: Rot, $g_4$-$g_7$: Refl)", fontsize=10)
    plt.ylabel("Test Success Rate (%)", fontsize=10)
    plt.ylim(0, 115)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"generalization_test_unseen_{size}.png", dpi=300)
    plt.close()
    print(f"  Saved plot: generalization_test_unseen_{size}.png")

def plot_density_robustness(results, size=13):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(7.5, 4.5))
    densities = [LOW_DENS, MED_DENS, HIGH_DENS]

    for label, res in results.items():
        means = [res["density_results"][d]["mean"] * 100 for d in densities]
        stds  = [res["density_results"][d]["std"] * 100 for d in densities]
        plt.errorbar(densities, means, yerr=stds, label=res["label"], color=res["color"],
                     linestyle=res["linestyle"], marker="o", capsize=5, lw=2)

    plt.xticks(densities, [f"Low ({LOW_DENS})", f"Medium ({MED_DENS})", f"High ({HIGH_DENS})"])
    plt.title(f"Robustness to Obstacle Density on Unseen Layouts ({size}x{size})", fontsize=12, fontweight="bold")
    plt.xlabel("Obstacle Count per Map", fontsize=10)
    plt.ylabel("Greedy Success Rate (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(f"density_robustness_{size}.png", dpi=300)
    plt.close()
    print(f"  Saved plot: density_robustness_{size}.png")

# ══════════════════════════════════════════════════════════════════════
#  Main Entrypoint
# ══════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    ("Proposed",                     D4EquivariantNet, "#E63946", "-",  2.0, False),
    ("Standard CNN + KL Guide",      StandardCNN,      "#457B9D", "--", 2.0, False),
    ("Standard CNN + 8x Aug",        StandardCNN,      "#FFB703", "-.", 2.0, True),
]

D4_LABELS = ["0° (id)", "90° CCW", "180°", "270° CCW",
             "Flip-H", "Flip-H+90°", "Flip-H+180°", "Flip-H+270°"]

def main():
    print("\n" + "="*75)
    print("  Starting Rigorous Validation Suite (Generalization on Unseen Layouts)")
    print(f"  Setup  : {len(SEEDS)} Seeds × {NUM_EPISODES} Episodes × {len(EXPERIMENTS)} Models")
    print(f"  Sizes  : {args.sizes}")
    print(f"  Device : {DEVICE}")
    print("="*75)

    global GRID_SIZE, MAX_STEPS, VAL_MAPS_20, TEST_MAPS_50, LOW_DENS, MED_DENS, HIGH_DENS, DENSITY_MAPS, GEN_MAPS_30
    
    t0 = time.time()

    for size in args.sizes:
        print(f"\n" + "#"*75)
        print(f"  Executing Experiments for Grid Size: {size}x{size}")
        print(f"#"*75)

        # Update global parameters dynamically
        GRID_SIZE = size
        MAX_STEPS = size * 3

        # Regenerate size-specific datasets
        VAL_MAPS_20 = generate_unseen_maps(20, int(GRID_SIZE * 0.9), 5555, GRID_SIZE)
        TEST_MAPS_50 = generate_unseen_maps(50, int(GRID_SIZE * 0.9), 9999, GRID_SIZE)

        LOW_DENS  = int(GRID_SIZE * 0.6)
        MED_DENS  = int(GRID_SIZE * 0.9)
        HIGH_DENS = int(GRID_SIZE * 1.2)

        DENSITY_MAPS = {
            LOW_DENS:  generate_unseen_maps(30, LOW_DENS, 8888, GRID_SIZE),
            MED_DENS:  generate_unseen_maps(30, MED_DENS, 8888, GRID_SIZE),
            HIGH_DENS: generate_unseen_maps(30, HIGH_DENS, 8888, GRID_SIZE),
        }
        GEN_MAPS_30 = generate_unseen_maps(30, int(GRID_SIZE * 0.9), 7777, GRID_SIZE)

        results = {}
        for label, cls, color, ls, beta_start, data_aug in EXPERIMENTS:
            res = run_experiment(label, cls, color, ls, beta_start, data_aug)
            results[label] = res

        # Save quantitative results for this size
        out_json = f"academic_results_rigorous_{size}.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved quantitative results: {out_json}")

        # Generate plots with size tags
        print(f"\nGenerating validation plots for {size}x{size}...")
        plot_learning_curves_unseen(results, size)
        plot_generalization_unseen(results, size)
        plot_density_robustness(results, size)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*75}")
    print(f"  All experiments completed successfully in {elapsed:.1f} minutes.")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
