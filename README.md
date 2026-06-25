# Advanced Grid-based Reinforcement Learning Theories & Implementations

A research-oriented repository detailing three mathematical formulations and complete PyTorch implementations for state-of-the-art grid-based reinforcement learning (RL):
1. **Equivariant Neural Networks (Dihedral $D_4$ Symmetry)**
2. **Actor-Critic MCTS Hybrid Search**
3. **Heuristic-Guided Policy Gradient via KL Divergence**

## Author & Affiliation
* **WonChan Cho**
* **Department of Mathematics, Sungkyunkwan University, Suwon, Republic of Korea**
* Email: `chln0124@skku.edu`

---

## Table of Contents
1. [Guide]
   - [1. Equivariant Neural Networks](#1-equivariant-neural-networks)
   - [2. Actor-Critic MCTS Integration](#2-actor-critic-mcts-integration)
   - [3. Heuristic-Guided Policy Gradient](#3-heuristic-guided-policy-gradient)
2. [Symmetric Grid Navigation Scenario](#symmetric-grid-navigation-scenario)
3. [Citation & Intellectual Property](#citation--intellectual-property)

---

# Guide

## 1. Equivariant Neural Networks
In 2D grid-based environments (e.g., board games or navigation maps), the grid possesses physical symmetries under rotation and reflection. The group representing these operations on a square grid is the **Dihedral Group $D_4$**, which contains 8 group elements (4 rotations and 4 reflections):
$$D_4 = \{ r_0, r_1, r_2, r_3, m_0, m_1, m_2, m_3 \}$$

### Mathematical Definition & Equivariance Property
A neural network function $f: X \to Y$ is said to be **equivariant** with respect to a symmetry group $G$ if transforming the input $x$ by a group element $g \in G$ yields the same result as transforming the output $f(x)$ by $g$:
$$f(g \cdot x) = g \cdot f(x) \quad \forall g \in G$$

For a policy network $\pi_\theta(a|s)$ predicting move probabilities over a grid:
$$\pi_\theta(g \cdot a \mid g \cdot s) = g \cdot \pi_\theta(a \mid s)$$

By enforcing equivariance directly into the neural network architecture, we constrain the hypothesis space to physically valid symmetrical functions, eliminating the need for 8x data augmentation, leading to **up to 8x faster training convergence**.

### Mathematical Proof of Equivariance
Let $s' = g \cdot s$. By the equivariance of the Equivariant Convolutional layers, the output features transform as $f_\theta(g \cdot s) = g \cdot f_\theta(s)$. Let $\mathbf{F} = f_\theta(s) \in \mathbb{R}^{|\mathcal{A}| \times H \times W}$ denote the feature maps. The action policy is obtained by taking the softmax of the feature map values at the agent's spatial position $p_{\text{agent}}$:
$$\pi_\theta(a \mid s) = \frac{\exp(\mathbf{F}_{a, p_{\text{agent}}})}{\sum_{a' \in \mathcal{A}} \exp(\mathbf{F}_{a', p_{\text{agent}}})}$$

When the state is transformed by $g$, the agent's position becomes $g \cdot p_{\text{agent}}$. The equivariant features at the new state are $[g \cdot \mathbf{F}]_{a, g \cdot p_{\text{agent}}}$. Because the channels of the equivariant convolutions transform according to the action permutation of $D_4$, we have $[g \cdot \mathbf{F}]_{g \cdot a, g \cdot p_{\text{agent}}} = \mathbf{F}_{a, p_{\text{agent}}}$. Applying the softmax over the action space yields:
$$\pi_\theta(g \cdot a \mid g \cdot s) = \frac{\exp([g \cdot \mathbf{F}]_{g \cdot a, g \cdot p_{\text{agent}}})}{\sum_{a' \in \mathcal{A}} \exp([g \cdot \mathbf{F}]_{g \cdot a', g \cdot p_{\text{agent}}})} = \frac{\exp(\mathbf{F}_{a, p_{\text{agent}}})}{\sum_{a' \in \mathcal{A}} \exp(\mathbf{F}_{a', p_{\text{agent}}})} = \pi_\theta(a \mid s) \quad \blacksquare$$

---

## 2. Actor-Critic MCTS Integration
Traditional Monte Carlo Tree Search (MCTS) relies on random playouts (rollouts) to evaluate leaf nodes, which exhibit high variance and consume massive computational time. In AlphaZero-like architectures, a dual-head network replaces rollouts with deep value evaluations.

### Formulation
A single neural network takes the board state $s$ as input and produces two outputs:
$$(\mathbf{p}, v) = f_\theta(s)$$
Where:
* $\mathbf{p} \in \mathbb{R}^{|A|}$ is the policy vector (prior probabilities $\pi(a|s)$).
* $v \in [-1, 1]$ is the scalar state value estimating the expected game outcome (Value).

### Search & Update Logic
During the selection phase, MCTS chooses actions maximizing the Upper Confidence Bound (UCB) variant (PUCT):
$$a_t = \arg\max_a \left( Q(s, a) + U(s, a) \right)$$
$$U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

Where $P(s, a) = \pi_\theta(a|s)$ is directly populated by the network's policy head. When a leaf node $s_L$ is expanded, instead of simulating to the end, it is immediately evaluated using the value head:
$$v = V_\theta(s_L)$$
The value $v$ is then backpropagated up the tree to update the action-value estimate $Q(s, a)$. For a single-agent MDP navigation task, value backpropagation does not alternate signs:
$$Q(s, a) \leftarrow Q(s, a) + \frac{v - Q(s, a)}{N(s, a)}$$

---

## 3. Heuristic-Guided Policy Gradient
Standard reinforcement learning starts with random exploration, which is highly inefficient. We can accelerate early-stage training and secure safety by steering the agent's policy toward established heuristics using an auxiliary loss function based on **Kullback-Leibler (KL) Divergence**.

### Loss Function
Let $\pi_\theta(a|s)$ be the trainable neural network policy, and $P_H(a|s)$ be a probability distribution derived from a heuristic rule. The total loss combines the standard Policy Gradient loss $L_{PG}(\theta)$, the value head MSE loss $L_V(\theta)$, and the heuristic regularization loss:
$$L(\theta) = L_{PG}(\theta) + \beta \cdot D_{KL}(P_H(s) \parallel \pi_\theta(s)) + \frac{1}{2} L_V(\theta)$$
where $L_{PG}(\theta) = - \frac{1}{B} \sum_{i=1}^B \log \pi_\theta(a_i | s_i) A_i$ and $D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \sum_{a \in A} P_H(a|s) \log \left( \frac{P_H(a|s)}{\pi_\theta(a|s)} \right)$.

### Theoretical Derivation of the Loss Gradient
Expanding the KL divergence term:
$$D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \sum_{a \in A} P_H(a|s) \log P_H(a|s) - \sum_{a \in A} P_H(a|s) \log \pi_\theta(a|s)$$

Since the heuristic $P_H(a|s)$ is independent of the network parameters $\theta$, taking the gradient with respect to $\theta$ yields:
$$\nabla_\theta D_{KL}(P_H(s) \parallel \pi_\theta(s)) = - \sum_{a \in A} P_H(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

Combining this with the policy gradient component $L_{PG}(\theta)$, we obtain the total policy parameter gradient:
$$\nabla_\theta L_{\text{policy}}(\theta) = - \frac{1}{B} \sum_{i=1}^B \left[ \mathbb{I}(a_i = a) A_i + \beta P_H(a | s_i) \right] \nabla_\theta \log \pi_\theta(a | s_i)$$

This shows that the heuristic policy distribution acts as a targeted pseudo-advantage, steering the policy updates towards safe movements during early exploration, proportional to $\beta P_H(a|s)$. The parameter $\beta \ge 0$ decays geometrically: $\beta_{t+1} = \max(\beta_t \cdot \gamma_{decay}, \beta_{min})$.

---

# Symmetric Grid Navigation Scenario

To demonstrate the generality of these core reinforcement learning concepts, this repository includes an **Autonomous Navigation & Obstacle Avoidance Simulator** on a 2D Grid map:
* **`Sample-Efficient Autonomous Navigation.../autonomous_env.py`**: A 13x13 grid environment with static obstacles. The agent (robot) moves in 8 directions (including diagonals) aiming to reach a destination coordinate safely.
* **`Sample-Efficient Autonomous Navigation.../train_navigation.py`**: Trains the agent using `D4EquivariantNet` (Policy-Value Network), `ActorCriticMCTS` (Search Tree), and `HeuristicGuidedLoss`. The baseline guidance is provided by a distance-based pathfinder heuristic ($P_H(s)$) which decays gradually to allow pure self-exploration.

### Running the Simulator:
```bash
cd "Sample-Efficient Autonomous Navigation using Group Equivariant Reinforcement Learning and Heuristic-Guided MCTS"
python run_academic_experiments.py
```

---

# Citation & Intellectual Property

If you use this work, theoretical formulations, or implementation code in your research or projects, please cite it as follows:

```bibtex
@misc{wonchan_cho_equivariant_guided_rl_2026,
  author       = {WonChan Cho},
  title        = {Advanced Grid-Based Reinforcement Learning Theories with Equivariant CNN and Heuristic-Guided Loss},
  institution  = {Department of Mathematics, Sungkyunkwan University},
  address      = {Suwon, Republic of Korea},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/WonC-Lab/WonC-Equivariant-Guided-RL}}
}
```

### License
This repository and all its theoretical derivations, mathematical formulations, and implementation codes are owned by **WonChan Cho**. They are licensed under the **MIT License**.
Copyright (c) 2026 WonChan Cho. All rights reserved.
