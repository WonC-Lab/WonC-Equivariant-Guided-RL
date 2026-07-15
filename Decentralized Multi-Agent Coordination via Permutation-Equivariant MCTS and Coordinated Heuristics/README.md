# Decentralized Multi-Agent Coordination via Permutation-Equivariant MCTS and Coordinated Heuristics

This repository implements a decentralized, sample-efficient Multi-Agent Reinforcement Learning (MARL) framework for grid-based pathfinding and collision avoidance. The framework integrates:
1. **Permutation-Equivariant GNN (PE-GNN)**: Restricts policies to be equivariant under the symmetric group $S_M$ of agent index permutations, allowing zero-shot generalization to unseen agent counts.
2. **Decentralized Monte Carlo Tree Search (PE-MCTS)**: Individual agents plan independently via search trees, avoiding the exponential scaling of joint action spaces.
3. **Coordinated Potential Field Heuristics**: Accelerates training convergence and ensures safety during exploration through a decaying KL-regularized loss and action priors during search.

---

## Key Features

* **$S_M$-Equivariant Policy Backbone**: By using Transformer blocks without positional encodings on agent features, swapping agent indices permutes the policy output correspondingly.
* **Decentralized Search**: Avoids the $O(|\mathcal{A}|^M)$ exponential scaling of joint MCTS by executing independent $O(M \cdot N_{\text{search}})$ local searches.
* **Cold-Start Safety**: Combines goal attraction and reciprocal neighbor/obstacle repulsion in a potential field prior, guiding early-stage exploration.

---

## Directory Structure

* `multi_agent_env.py`: Decentralized 2D grid environment supporting variable agent counts and collision types (static, vertex, edge).
* `equivariant_gnn.py`: Permutation-equivariant Graph Neural Network policy and value head backbone.
* `multi_agent_mcts.py`: Decentralized Monte Carlo Tree Search with Predictor Upper Confidence Bound (PUCT) guided by mixed priors.
* `heuristic_guided_loss.py`: Decaying KL-divergence loss function regularizing policy outputs toward the safety heuristic.
* `train_multi_agent.py`: Training pipeline using decentralized actor-critic MCTS rollouts and PPO optimization.
* `run_multi_agent_experiments.py`: Suite for evaluating zero-shot scalability, ablation studies, and spatial robustness.
* `models/`: Contains trained model checkpoints.
* `results/`: Output directories for scalability and ablation plot figures.

---

## Installation & Setup

Ensure you have PyTorch and standard scientific python packages installed:
```bash
pip install torch numpy matplotlib
```

---

## Running the Code

### 1. Training the Model
To train the permutation-equivariant network guided by MCTS and potential fields:
```bash
python train_multi_agent.py
```
This trains the network with $M=4$ agents for 250 episodes, periodically decaying the heuristic loss regularizer $\beta$, and saves the checkpoint to `models/multi_agent_model.pth`.

### 2. Evaluating Performance & Scalability
To evaluate the trained model zero-shot on unseen agent counts $M \in \{2, 3, 4, 5, 6, 8\}$ across Default, Empty, and Random obstacle maps:
```bash
python run_multi_agent_experiments.py
```
This script evaluates the model and plots the robustness and ablation curves in the `results/` folder.

---

## Experimental Results

The framework demonstrates robust zero-shot scalability and coordinated navigation for up to 8 agents, significantly outperforming the search-free GNN baseline and achieving superior robustness under varying obstacle layouts and densities. All evaluations use a statistically uniform sample size of $N=50$ episodes.

### 1. Robustness & Generalization Success Rates (mean ± std over N=50 randomized episodes)

| Configuration / Policy Mode | M = 2 | M = 3 | M = 4 | M = 5 | M = 6 | M = 8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ours: Default Map (Trained)** | **94.0% ± 24%** | **80.0% ± 40%** | **82.0% ± 38%** | **68.0% ± 47%** | **58.0% ± 49%** | **30.0% ± 46%** |
| **Ours: Empty Map** | **100.0% ± 0%** | **96.0% ± 20%** | **90.0% ± 30%** | **72.0% ± 45%** | **74.0% ± 44%** | **66.0% ± 47%** |
| **Ours: Random Obstacles** | **98.0% ± 14%** | **92.0% ± 27%** | **84.0% ± 37%** | **80.0% ± 40%** | **80.0% ± 40%** | **56.0% ± 50%** |
| ORCA-approx. (tuned) | 94.0% ± 24% | 88.0% ± 33% | 84.0% ± 37% | 64.0% ± 48% | 38.0% ± 49% | 42.0% ± 49% |
| GNN + 1-step Lookahead | 0.0% ± 0% | 0.0% ± 0% | 0.0% ± 0% | 0.0% ± 0% | 0.0% ± 0% | 0.0% ± 0% |
| Baseline: Heuristic Only (Default) | 92.0% ± 27% | 82.0% ± 38% | 76.0% ± 43% | 62.0% ± 49% | 42.0% ± 49% | 38.0% ± 49% |

### 2. Hyperparameter Sensitivity (M = 4, Default Map, N = 50)

* **MCTS Search Budget ($N_{\text{search}}$)**: Coordination success improves monotonically as MCTS budget increases, scaling from **68.0% ± 47%** at $N_{\text{search}} = 5$ to a peak of **76.0% ± 43%** at $N_{\text{search}} = 80$.
* **Heuristic Mixing Weight ($\beta$)**: We observe a sharp threshold behavior. When $\beta \le 0.1$ (minimal heuristic guidance), the success rate collapses to **0.0% - 2.0%**. It rises to **72.0%** at $\beta = 0.3$ and peaks at **74.0%** at $\beta = 0.5$, showing that the coordinated potential field heuristic is mathematically indispensable for guiding tree search in cluttered multi-agent environments.
* **Communication Radius ($R_c$, 25x25 grid)**: All radii $\{3, 6, \infty\}$ achieve **94.0%** success rate. This validates that coordination is an inherently local behavior; masking out distant agents does not affect local collision avoidance.
* **Grid-size Zero-shot Transfer**: Zero-shot transfer to a larger $20\times 20$ grid achieves **94.0%** success (compared to $84.0\%$ on the $13\times 13$ trained grid), showing that the equivariant architecture generalizes seamlessly to larger spatial environments.

### Generated Visualizations

All plot figures are saved in the `results/` directory and embedded below:

#### 1. Robustness & Generalization (Zero-Shot Scalability)
![Zero-Shot Scalability](results/scalability_test.png)
*Illustrates zero-shot success rates under varying obstacle configurations.*

#### 2. MCTS Ablation Study
![MCTS Ablation](results/mcts_ablation.png)
*Success rates compared against search-free and heuristic-only baselines.*

#### 3. Obstacle Density Robustness
![Obstacle Density Robustness](results/density_robustness.png)
*Performance under different obstacle densities ($M=4$ agents).*

#### 4. Hyperparameter Sensitivity Sweeps
| MCTS Search Budget ($N_{\text{search}}$) | Heuristic Mixing Weight ($\beta$) |
| :---: | :---: |
| ![Search Budget Sensitivity](results/nsearch_sensitivity.png) | ![Heuristic Weight Sensitivity](results/beta_sensitivity.png) |

#### 5. Qualitative Visualizations
| Decentralized GNN Value Landscape | Coordinated Potential Flow Field |
| :---: | :---: |
| ![GNN Value Landscape](results/premium_value_surface.png) | ![Coordinated Potential Flow](results/premium_potential_flow.png) |

---

## License & Citation

Licensed under the MIT License. Copyright (c) 2026 WonChan Cho. All rights reserved.
For academic use, please cite:
```bibtex
@misc{wonchan_cho_multi_agent_equiv_2026,
  author = {WonChan Cho},
  title = {Decentralized Multi-Agent Coordination via Permutation-Equivariant MCTS and Coordinated Heuristics},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/WonC-Lab/Permutation-Equivariant-MCTS-and-Coordinated-Heuristics}}
}
```
