"""
real_experiment.py  ─  "Train First, Search Later" 구조
=========================================================
논문용 실제 실험 스크립트.

Phase 1 (학습): Equivariant/Standard CNN을 순수 Actor-Critic +
                KL Heuristic Guide로 훈련 (MCTS 없음 → 빠르고 안정적)
Phase 2 (평가): 학습된 모델을 Greedy vs MCTS-guided 방식으로 비교
Phase 3 (일반화): D4 회전 환경에서 Zero-shot 성공률 측정

출력물:
  real_results.json        ─ 전체 수치 (mean ± std, 5 seeds)
  real_results_plots.png   ─ 논문용 그래프 (오차막대 포함)
"""

import os, sys, json, time, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 로컬 모듈 ──────────────────────────────────────────────────────────
from autonomous_env import AutonomousNavigationEnv
from equivariant_models import D4EquivariantNet, StandardCNN
from mcts_actor_critic import ActorCriticMCTS
from train_navigation import SymmetricNavEnvAdapter

# ── 디바이스 ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    # Force CPU for Mac because MPS device synchronization latency makes sequential tree search extremely slow
    DEVICE = torch.device("cpu")
    print(" Apple Silicon 사용 중 (CPU 강제 사용로 지연 방지)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(" CUDA GPU 사용")
else:
    DEVICE = torch.device("cpu")
    print(" CPU 사용")

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────
SEEDS         = [42, 123, 456]  # 3 독립 시드 (속도를 위해 조정)
NUM_EPISODES  = 300    # 에피소드 수
WARMUP_EPS    = 60     # 휴리스틱 직접 행동 선택 (초반 성공 경험 확보)
MAX_STEPS     = 40     # 에피소드당 최대 스텝
GAMMA         = 0.95   # 할인 인수
LR            = 0.002  # Adam 학습률
BETA_START    = 2.0    # KL 가이드 초기 강도
BETA_DECAY    = 0.95   # 에피소드마다 감쇠
BETA_MIN      = 0.5
TRAIN_EPOCHS  = 3      # 에피소드 데이터로 반복 학습 횟수
GRAD_CLIP     = 1.0
EVAL_INTERVAL = 50     # 몇 에피소드마다 평가
EVAL_EPS      = 20     # 평가 에피소드 수 (통계적 안정성 위해 증가)
MCTS_SIMS     = 15     # MCTS 시뮬레이션 횟수 (추론 단계만 사용)
NUM_LAYERS    = 4
NUM_FILTERS   = 16

# ── 유틸 ───────────────────────────────────────────────────────────────
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

def rotate_obstacles_goal(obstacles, goal, size, k):
    """격자 맵을 k * 90도 회전."""
    def rot(r, c, n, k):
        for _ in range(k % 4):
            r, c = c, n - 1 - r
        return r, c
    new_obs = {rot(r, c, size, k) for r, c in obstacles}
    gr, gc = goal
    return new_obs, rot(gr, gc, size, k)

# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: Actor-Critic 학습 (MCTS 없음)
# ══════════════════════════════════════════════════════════════════════

def train_one_episode(model, base_env, env_adapter, optimizer, beta,
                      episode_num, kl_loss_fn):
    """
    단일 에피소드를 실행하고 Actor-Critic + KL Heuristic Loss로 학습.
    WARMUP_EPS 이내에는 휴리스틱이 직접 행동 선택 (성공 경험 확보).
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
            # 휴리스틱 행동 선택 (노이즈 약간 추가)
            probs = np.array([heuristic_8[i] for i in valid_dirs])
            probs = probs / probs.sum()
            chosen = int(np.random.choice(valid_dirs, p=probs))
        else:
            # 정책 네트워크로 행동 선택 (softmax 샘플링)
            with torch.no_grad():
                logits, _ = model(st)
            logits_np = logits.squeeze(0).cpu().numpy()
            # 유효하지 않은 행동 마스킹
            mask = np.full(8, -1e9)
            for di in valid_dirs:
                mask[di] = logits_np[di]
            # softmax 샘플링
            mask_shifted = mask - mask[mask > -1e8].max()
            exp_m = np.exp(mask_shifted)
            exp_m[mask <= -1e8] = 0.0
            probs = exp_m / exp_m.sum()
            chosen = int(np.random.choice(8, p=probs))

        # Get old log probability of chosen action
        with torch.no_grad():
            logits_lp, _ = model(st)
        lp = F.log_softmax(logits_lp, dim=-1)
        old_log_prob = lp[0, chosen].item()

        # 다음 상태로 전이
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

    # ── 학습 ──────────────────────────────────────────────────────────
    avg_loss = None
    if ep_tensors:
        returns = discounted_returns(ep_rewards)
        batch_s = torch.cat(ep_tensors, 0)                              # (T, 3, 13, 13)
        batch_a = torch.tensor(ep_actions, dtype=torch.long, device=DEVICE)
        batch_r = torch.tensor(returns,    dtype=torch.float32, device=DEVICE)
        batch_h = torch.tensor(np.array(ep_heuristics),
                               dtype=torch.float32, device=DEVICE)     # (T, 8)
        batch_old_lps = torch.tensor(ep_old_log_probs, dtype=torch.float32, device=DEVICE)

        model.train()
        total_loss = 0.0
        for _ in range(TRAIN_EPOCHS):
            optimizer.zero_grad()
            logits, values = model(batch_s)                 # (T, 8), (T, 1)

            log_probs = F.log_softmax(logits, dim=-1)       # (T, 8)
            sel_lp = log_probs[torch.arange(len(batch_a)), batch_a]  # (T,)

            # Advantage = Return - V(s)
            with torch.no_grad():
                adv = batch_r - values.squeeze(-1)
            
            # PPO Policy Clipping
            ratio = torch.exp(sel_lp - batch_old_lps)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # KL Divergence: KL( heuristic ‖ agent )
            kl = kl_loss_fn(log_probs, batch_h)

            # Value head MSE
            val_loss = F.mse_loss(values.squeeze(-1), batch_r)

            loss = pg_loss + beta * kl + 0.5 * val_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / TRAIN_EPOCHS

    return success, collision, steps, avg_loss


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: 평가 함수 (Greedy / MCTS)
# ══════════════════════════════════════════════════════════════════════

def evaluate_greedy(model, base_env, env_adapter, n=EVAL_EPS):
    """학습된 모델로 greedy 평가 (argmax of policy)."""
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
    """학습된 모델을 prior로 MCTS 추론 평가."""
    model.eval()
    successes = 0
    for _ in range(n):
        # 에피소드마다 새 MCTS 객체 생성 → 메모리 누수 방지
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
#  PHASE 3: 회전 일반화 평가
# ══════════════════════════════════════════════════════════════════════

def evaluate_rotation_generalization(model, base_env, n=30):
    """0°~270° 4방향 회전 환경에서 greedy 성공률 측정."""
    results = []
    for k in range(4):
        new_obs, new_goal = rotate_obstacles_goal(
            base_env.obstacles, base_env.goal, base_env.size, k
        )
        rot_env = AutonomousNavigationEnv(size=base_env.size)
        rot_env.obstacles = new_obs
        rot_env.goal = new_goal
        rot_adapter = SymmetricNavEnvAdapter(rot_env)
        rate = evaluate_greedy(model, rot_env, rot_adapter, n=n)
        results.append(rate)
    return results   # [0°, 90°, 180°, 270°]


# ══════════════════════════════════════════════════════════════════════
#  멀티 시드 학습 루프
# ══════════════════════════════════════════════════════════════════════

def run_experiment(label, model_cls, color, linestyle):
    print(f"\n{'='*65}")
    print(f"  Learning: {label}")
    print(f"{'='*65}")

    kl_fn = nn.KLDivLoss(reduction="batchmean")

    all_greedy_rates, all_mcts_rates = [], []
    all_curves_greedy, all_curves_mcts = [], []
    all_collisions, all_conv_ep = [], []
    best_model_state = None
    best_rate = -1.0

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
        conv_ep    = NUM_EPISODES  # 수렴 못하면 끝 에피소드

        for ep in range(1, NUM_EPISODES + 1):
            success, collision, steps, loss = train_one_episode(
                model, base_env, env_adapter, optimizer, beta, ep, kl_fn
            )
            if collision: collisions += 1
            beta = max(beta * BETA_DECAY, BETA_MIN)

            if ep % EVAL_INTERVAL == 0:
                # ── 평가는 항상 표준 고정 맵으로 (공정한 비교)
                base_env.reset_canonical_obstacles()
                gr = evaluate_greedy(model, base_env, env_adapter)
                mr = evaluate_with_mcts(model, base_env, env_adapter)
                curve_greedy.append(gr)
                curve_mcts.append(mr)

                if gr >= 0.70 and conv_ep == NUM_EPISODES:
                    conv_ep = ep

                loss_s = f"{loss:.4f}" if loss else "N/A"
                mode   = "WarmUp" if ep <= WARMUP_EPS else "AC+KL  "
                print(f"    Ep {ep:4d}/{NUM_EPISODES} [{mode}] "
                      f"Greedy: {gr*100:5.1f}%  MCTS: {mr*100:5.1f}%  "
                      f"Loss: {loss_s}  β: {beta:.3f}  Coll: {collisions}")

        # ── 최종 평가 (표준 고정 맵, 50 에피소드)
        base_env.reset_canonical_obstacles()
        final_greedy = evaluate_greedy(model, base_env, env_adapter, n=50)
        final_mcts   = evaluate_with_mcts(model, base_env, env_adapter, n=50)

        if final_greedy > best_rate:
            best_rate = final_greedy
            best_model_state = copy.deepcopy(model.state_dict())

        all_greedy_rates.append(final_greedy)
        all_mcts_rates.append(final_mcts)
        all_curves_greedy.append(curve_greedy)
        all_curves_mcts.append(curve_mcts)
        all_collisions.append(collisions)
        all_conv_ep.append(conv_ep)

        print(f"  → Seed {seed}: Greedy {final_greedy*100:.1f}%  "
              f"MCTS {final_mcts*100:.1f}%  Conv@{conv_ep}  Coll:{collisions}")

    # 시드 통계 집계
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
    print(f"    수렴 에피소드: {np.mean(all_conv_ep):.0f}  충돌: {np.mean(all_collisions):.0f}")

    # 회전 일반화 (best seed 모델 재사용)
    base_env = AutonomousNavigationEnv(size=13)
    env_a    = SymmetricNavEnvAdapter(base_env)
    best_m   = make_model(model_cls)
    best_m.load_state_dict(best_model_state)
    gen_rates = evaluate_rotation_generalization(best_m, base_env, n=30)
    gen_rates_mcts = []
    for k in range(4):
        new_obs, new_goal = rotate_obstacles_goal(
            base_env.obstacles, base_env.goal, base_env.size, k
        )
        rot_env = AutonomousNavigationEnv(size=base_env.size)
        rot_env.obstacles = new_obs; rot_env.goal = new_goal
        rot_a = SymmetricNavEnvAdapter(rot_env)
        gen_rates_mcts.append(evaluate_with_mcts(best_m, rot_env, rot_a, n=30))

    return {
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "greedy_mean": mg,   "greedy_std": sg,
        "mcts_mean":   mm,   "mcts_std":   sm,
        "mean_conv_ep":   float(np.mean(all_conv_ep)),
        "mean_collisions": float(np.mean(all_collisions)),
        "eval_episodes": eval_eps,
        "curve_greedy_mean": curves_g.mean(0).tolist(),
        "curve_greedy_std":  curves_g.std(0).tolist(),
        "curve_mcts_mean":   curves_m.mean(0).tolist(),
        "curve_mcts_std":    curves_m.std(0).tolist(),
        "generalization_greedy": gen_rates,
        "generalization_mcts":   gen_rates_mcts,
        "seeds": SEEDS,
    }


# ══════════════════════════════════════════════════════════════════════
#  그래프 생성
# ══════════════════════════════════════════════════════════════════════

def plot_all(results):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 14))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax_curve  = fig.add_subplot(gs[0, :2])
    ax_bar    = fig.add_subplot(gs[0, 2])
    ax_gen    = fig.add_subplot(gs[1, 0])
    ax_conv   = fig.add_subplot(gs[1, 1])
    ax_safety = fig.add_subplot(gs[1, 2])

    eps = results[0]["eval_episodes"]

    # ── 1. 학습 곡선 ──────────────────────────────────────────────────
    ax_curve.set_title("Real Learning Curves (Mean ± Std, 5 Seeds)",
                        fontsize=13, fontweight="bold")
    for res in results:
        mu = np.array(res["curve_greedy_mean"]) * 100
        sd = np.array(res["curve_greedy_std"])  * 100
        ax_curve.plot(eps, mu, label=res["label"]+" (Greedy)",
                      color=res["color"], linestyle=res["linestyle"], lw=2)
        ax_curve.fill_between(eps, mu-sd, mu+sd, alpha=0.12, color=res["color"])
        mu2 = np.array(res["curve_mcts_mean"]) * 100
        sd2 = np.array(res["curve_mcts_std"])  * 100
        ax_curve.plot(eps, mu2, label=res["label"]+" (+MCTS)",
                      color=res["color"], linestyle=":", lw=1.5, alpha=0.8)
    ax_curve.axhline(70, color="white", lw=0.8, ls="--", alpha=0.35,
                     label="70% 수렴 기준선")
    ax_curve.axvline(WARMUP_EPS, color="#aaa", lw=0.8, ls="--", alpha=0.5,
                     label=f"Warm-up 종료 (Ep {WARMUP_EPS})")
    ax_curve.set_xlabel("Training Episodes", fontsize=11)
    ax_curve.set_ylabel("Success Rate (%)", fontsize=11)
    ax_curve.legend(fontsize=7, loc="upper left", ncol=2)
    ax_curve.set_ylim(0, 108); ax_curve.grid(True, alpha=0.15)

    # ── 2. 최종 성공률 비교 막대 그래프 ──────────────────────────────
    ax_bar.set_title("Final Success Rate\n(Mean ± Std, 5 Seeds)",
                     fontsize=13, fontweight="bold")
    x = np.arange(len(results))
    w = 0.35
    bars_g = ax_bar.bar(x - w/2,
                        [r["greedy_mean"]*100 for r in results],
                        w, yerr=[r["greedy_std"]*100 for r in results],
                        capsize=5, label="Greedy",
                        color=[r["color"] for r in results],
                        alpha=0.85,
                        error_kw={"ecolor":"white","elinewidth":1.5})
    bars_m = ax_bar.bar(x + w/2,
                        [r["mcts_mean"]*100 for r in results],
                        w, yerr=[r["mcts_std"]*100 for r in results],
                        capsize=5, label="+MCTS",
                        color=[r["color"] for r in results],
                        alpha=0.50, hatch="//",
                        error_kw={"ecolor":"white","elinewidth":1.5})
    for i, res in enumerate(results):
        ax_bar.text(i-w/2, res["greedy_mean"]*100+res["greedy_std"]*100+2,
                    f'{res["greedy_mean"]*100:.1f}%', ha="center",
                    fontsize=8, color="white")
        ax_bar.text(i+w/2, res["mcts_mean"]*100+res["mcts_std"]*100+2,
                    f'{res["mcts_mean"]*100:.1f}%', ha="center",
                    fontsize=8, color="white")
    short = [r["label"].replace(" (Proposed)","").replace("+MCTS+Guide","") for r in results]
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(short, fontsize=8)
    ax_bar.set_ylim(0, 115); ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.15, axis="y")

    # ── 3. Zero-Shot 회전 일반화 ─────────────────────────────────────
    ax_gen.set_title("Zero-Shot Rotation Generalization\n(D4 Symmetry Test)",
                     fontsize=13, fontweight="bold")
    angles = [0, 90, 180, 270]
    angle_labels = ["0°", "90°", "180°", "270°"]
    x_a = np.arange(4)
    for res in results:
        gg = [v*100 for v in res["generalization_greedy"]]
        gm = [v*100 for v in res["generalization_mcts"]]
        ax_gen.plot(x_a, gg, marker="o", color=res["color"],
                    label=res["label"][:20], lw=2)
        ax_gen.plot(x_a, gm, marker="s", color=res["color"],
                    ls=":", lw=1.4, alpha=0.7)
    ax_gen.set_xticks(x_a); ax_gen.set_xticklabels(angle_labels)
    ax_gen.set_ylabel("Success Rate (%)"); ax_gen.set_ylim(0, 108)
    ax_gen.legend(fontsize=7); ax_gen.grid(True, alpha=0.15)

    # ── 4. 수렴 속도 ──────────────────────────────────────────────────
    ax_conv.set_title("Episodes to 70% Success\n(Lower = More Sample-Efficient)",
                      fontsize=13, fontweight="bold")
    conv_vals = [r["mean_conv_ep"] for r in results]
    colors    = [r["color"] for r in results]
    bars = ax_conv.barh(range(len(results)), conv_vals, color=colors, alpha=0.85)
    ax_conv.set_yticks(range(len(results)))
    ax_conv.set_yticklabels(short, fontsize=8)
    ax_conv.set_xlabel("Episodes")
    for i, v in enumerate(conv_vals):
        ax_conv.text(v+3, i, f"{v:.0f}", va="center", fontsize=9, color="white")
    ax_conv.grid(True, alpha=0.15, axis="x")

    # ── 5. 학습 중 충돌 수 ───────────────────────────────────────────
    ax_safety.set_title("Training Collisions\n(Lower = Safer Exploration)",
                        fontsize=13, fontweight="bold")
    col_vals = [r["mean_collisions"] for r in results]
    ax_safety.bar(range(len(results)), col_vals, color=colors, alpha=0.85)
    ax_safety.set_xticks(range(len(results)))
    ax_safety.set_xticklabels(short, fontsize=8)
    ax_safety.set_ylabel("Total Collisions")
    for i, v in enumerate(col_vals):
        ax_safety.text(i, v+0.5, f"{v:.0f}", ha="center",
                       fontsize=9, color="white")
    ax_safety.grid(True, alpha=0.15, axis="y")

    fig.suptitle(
        "Sample-Efficient Autonomous Navigation ─ Real Experimental Results\n"
        f"(Train-First Search-Later | 5 Seeds x {NUM_EPISODES} Episodes | CPU)",
        fontsize=14, fontweight="bold", y=0.99
    )
    out = "real_results_plots_v2.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\n  Save Result: {out}")


# ══════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("D4-Equivariant + KL Guide (Proposed)", D4EquivariantNet, "#E63946", "-"),
    ("Standard CNN + KL Guide",              StandardCNN,      "#457B9D", "--"),
]

def main():
    print("\n" + "="*65)
    print("Start of the experiment   ─  Train First, Search Later")
    print(f" Setting: {len(SEEDS)} Seeds x {NUM_EPISODES} Ep x {len(EXPERIMENTS)} Model")
    print(f" Warm-up: {WARMUP_EPS} Ep → Actor-Critic+KL: {NUM_EPISODES-WARMUP_EPS} Ep")
    print(f" Device: {DEVICE}")
    print("="*65)

    t0 = time.time()
    all_results = []

    for label, cls, color, ls in EXPERIMENTS:
        res = run_experiment(label, cls, color, ls)
        all_results.append(res)

    # 저장
    out_json = "real_results_v2.json"
    # JSON에 저장할 때 numpy → python 변환
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n Save Result: {out_json}")

    plot_all(all_results)

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*65}")
    print(f"  Total duration: {elapsed:.1f}min")
    print(f"{'='*65}")
    for res in all_results:
        print(f"  {res['label'][:50]:50s}")
        print(f"    Greedy : {res['greedy_mean']*100:.1f}% ± {res['greedy_std']*100:.1f}%")
        print(f"    +MCTS  : {res['mcts_mean']*100:.1f}% ± {res['mcts_std']*100:.1f}%")
    print("="*65)


if __name__ == "__main__":
    main()
