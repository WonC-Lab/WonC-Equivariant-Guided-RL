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
     - [Group Definition & Actions](#group-definition--actions)
     - [Equivariance Property](#equivariance-property)
     - [Rigorous Proofs of Equivariance](#rigorous-proofs-of-equivariance)
   - [2. Actor-Critic MCTS Integration](#2-actor-critic-mcts-integration)
     - [Formulation](#formulation)
     - [Search & Update Logic](#search--update-logic)
     - [Asymptotic Convergence Analysis](#asymptotic-convergence-analysis)
   - [3. Heuristic-Guided Policy Gradient](#3-heuristic-guided-policy-gradient)
     - [Loss Function](#loss-function)
     - [Theoretical Derivation of the Loss Gradient](#theoretical-derivation-of-the-loss-gradient)
     - [PPO Clipped Objective & Gradient Derivation](#ppo-clipped-objective--gradient-derivation)
2. [Symmetric Grid Navigation Scenario](#symmetric-grid-navigation-scenario)
3. [Citation & Intellectual Property](#citation--intellectual-property)

---

# Guide

## 1. Equivariant Neural Networks

In 2D grid-based environments (e.g., board games or navigation maps), the state space possesses physical symmetries under rotation and reflection. The group representing these operations on a square grid is the **Dihedral Group $D_4$**, which is the isometry group of a square.

### Group Definition & Actions
Formally, $D_4$ is defined by the presentation:
$D_4 = \langle r, m \mid r^4 = e, m^2 = e, mrm = r^{-1} \rangle = \{ r^k m^j \mid k \in \{0, 1, 2, 3\}, j \in \{0, 1\} \}$
where $r$ represents a $90^\circ$ counter-clockwise rotation, and $m$ represents a horizontal reflection.

Let the state space $S$ be represented as the space of square grid configurations $L^2(\mathcal{G}, \mathbb{R}^C) \cong \mathbb{R}^{C \times H \times W}$, where $\mathcal{G} \subset \mathbb{Z}^2$ is the grid coordinates, $C$ is the number of channels, and $H, W$ are the height and width of the grid. The action of a group element $g \in D_4$ on a grid coordinate $p \in \mathcal{G}$ is denoted by $g \cdot p$. We define the input representation (group action) $\rho_{\text{in}}(g)$ on a state $s \in S$ as:
$$\left[\rho_{\text{in}}(g) \cdot s\right] (p) = \rho_{\text{chan}}(g) s(g^{-1} \cdot p)$$
where $\rho_{\text{chan}}(g) \in \mathbb{R}^{C \times C}$ represents how the channel dimensions transform under the group element $g$.

Let $\mathcal{A}$ represent the action space (e.g., 8-directional movement directions). The action of $D_4$ on $\mathcal{A}$ is given by a permutation representation $\rho_{\text{out}}(g) \in \mathbb{R}^{|\mathcal{A}| \times |\mathcal{A}|}$ which maps each action $a \in \mathcal{A}$ to $g \cdot a$.

### Equivariance Property
A neural network policy function $\pi_\theta: S \to \mathcal{P}(\mathcal{A})$ maps a state $s$ to a probability distribution over the action space $\mathcal{A}$. The network is defined as **equivariant** with respect to the group $D_4$ if:
$$\pi_\theta(\rho_{\text{in}}(g) \cdot s) = \rho_{\text{out}}(g) \cdot \pi_\theta(s) \quad \forall g \in D_4, s \in S$$
Evaluating this element-wise over individual actions:
$$\pi_\theta(g \cdot a \mid \rho_{\text{in}}(g) \cdot s) = \pi_\theta(a \mid s) \quad \forall g \in D_4, s \in S, a \in \mathcal{A}$$

By enforcing equivariance directly into the neural network architecture (rather than relying on data augmentation), the hypothesis space is constrained strictly to physically valid symmetrical functions, resulting in **up to 8x faster training convergence** and enhanced generalization.

### Rigorous Proofs of Equivariance

#### Theorem 1: Equivariance of Group Convolutional Layers
Let $G$ be a discrete group. A group convolutional layer mapping an input feature map $f: G \to \mathbb{R}^{C_{\text{in}}}$ to an output feature map $f * \psi: G \to \mathbb{R}^{C_{\text{out}}}$ using a kernel $\psi: G \to \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$ is defined as:
$$\left[f * \psi\right] (g) = \sum_{h \in G} \psi(h^{-1} g) f(h)$$
Let $\rho_L(g')$ be the left regular representation acting on $f$, defined by $\left[\rho_L(g') \cdot f\right] (g) = f(g'^{-1} g)$. Then the group convolution operation is equivariant with respect to $\rho_L$:
$$\left[(\rho_L(g') \cdot f) * \psi\right] = \rho_L(g') \cdot \left[f * \psi\right]$$

*Proof:*
Expanding the definition of group convolution under the transformed input $\rho_L(g') \cdot f$:
$$\left[(\rho_L(g') \cdot f) * \psi\right] (g) = \sum_{h \in G} \psi(h^{-1} g) \left[\rho_L(g') \cdot f\right] (h) = \sum_{h \in G} \psi(h^{-1} g) f(g'^{-1} h)$$
Letting $h' = g'^{-1} h$, which implies $h = g' h'$. Since $h \mapsto g' h'$ is a bijection on the group $G$, we can substitute the summation index:
$$\left[(\rho_L(g') \cdot f) * \psi\right] (g) = \sum_{h' \in G} \psi((g' h')^{-1} g) f(h') = \sum_{h' \in G} \psi(h'^{-1} g'^{-1} g) f(h')$$
By definition of group convolution evaluated at the element $g'^{-1}g$, we obtain:
$$= \left[f * \psi\right] (g'^{-1} g) = \left[\rho_L(g') \cdot (f * \psi)\right] (g)$$
Thus, $\left[(\rho_L(g') \cdot f) * \psi\right] = \rho_L(g') \cdot \left[f * \psi\right]$ holds for all $g \in G$, proving that group convolution is equivariant under the group action. $\blacksquare$

#### Theorem 2: Equivariance of the Output Policy under Softmax
Let $\mathbf{F}_\theta(s) \in \mathbb{R}^{|\mathcal{A}| \times H \times W}$ be the final spatial feature map output by the equivariant layers. These features satisfy the spatial equivariance relation:
$\mathbf{F}_\theta(\rho_{\text{in}}(g) \cdot s)_{g \cdot a, g \cdot p} = \mathbf{F}_\theta(s)_{a, p}$
Applying a softmax over the action dimension at the agent's spatial position $p_{\text{agent}}$ defines the policy distribution $\pi_\theta(a|s)$. Then the policy $\pi_\theta$ is equivariant under the spatial transformations.

*Proof:*
Let $s' = \rho_{\text{in}}(g) \cdot s$. The policy at state $s'$ evaluated for a transformed action $g \cdot a$ at the transformed position $g \cdot p_{\text{agent}}$ is given by:
$\pi_\theta(g \cdot a \mid \rho_{\text{in}}(g) \cdot s) = \frac{\exp\left(\mathbf{F}_\theta(\rho_{\text{in}}(g) \cdot s)_{g \cdot a, g \cdot p_{\text{agent}}}\right)}{\sum_{a' \in \mathcal{A}} \exp\left(\mathbf{F}_\theta(\rho_{\text{in}}(g) \cdot s)_{g \cdot a', g \cdot p_{\text{agent}}}\right)}$
Using the spatial equivariance property of the feature maps, we substitute the values:
$\pi_\theta(g \cdot a \mid \rho_{\text{in}}(g) \cdot s) = \frac{\exp\left(\mathbf{F}_\theta(s)_{a, p_{\text{agent}}}\right)}{\sum_{a' \in \mathcal{A}} \exp\left(\mathbf{F}_\theta(s)_{g^{-1} \cdot a', p_{\text{agent}}}\right)}$
Because the group action $a' \mapsto g^{-1} \cdot a'$ is a bijection over the finite action space $\mathcal{A}$, summing over all $a' \in \mathcal{A}$ is equivalent to summing over all $b = g^{-1} \cdot a' \in \mathcal{A}$:
$\sum_{a' \in \mathcal{A}} \exp\left(\mathbf{F}_\theta(s)_{g^{-1} \cdot a', p_{\text{agent}}}\right) = \sum_{b \in \mathcal{A}} \exp\left(\mathbf{F}_\theta(s)_{b, p_{\text{agent}}}\right)$
Substituting this back into the denominator:
$\pi_\theta(g \cdot a \mid \rho_{\text{in}}(g) \cdot s) = \frac{\exp\left(\mathbf{F}_\theta(s)_{a, p_{\text{agent}}}\right)}{\sum_{b \in \mathcal{A}} \exp\left(\mathbf{F}_\theta(s)_{b, p_{\text{agent}}}\right)} = \pi_\theta(a \mid s) \quad \blacksquare$

---

## 2. Actor-Critic MCTS Integration

Traditional Monte Carlo Tree Search (MCTS) relies on random rollout simulations to evaluate leaf nodes, which exhibit high variance and consume massive computational resources. AlphaZero-like architectures replace these rollouts with deep policy-value neural networks to guide tree expansions.

### Formulation
A dual-head neural network takes the board state $s$ as input and outputs both policy and value estimations:
$(\mathbf{p}, v) = f_\theta(s)$
where:
* $\mathbf{p} \in \mathcal{P}(\mathcal{A})$ is the prior probability distribution $\pi_\theta(a|s)$ over actions.
* $v \in [-1, 1]$ is the scalar state value estimating the expected long-term outcome.

### Search & Update Logic
During the selection phase, MCTS navigates from the root node by choosing actions that maximize the Predictor Upper Confidence Bound (PUCT) variant:
$a_t = \arg\max_{a \in \mathcal{A}} \left( Q(s, a) + U(s, a) \right)$
$U(s, a) = c(s) \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$
where:
* $Q(s, a)$ is the mean action-value.
* $P(s, a) = \pi_\theta(a|s)$ is the prior probability of action $a$.
* $N(s, a)$ is the visit count of the edge $(s, a)$, and $N(s) = \sum_{b \in \mathcal{A}} N(s, b)$.
* $c(s)$ is a dynamically scaled exploration factor:
  $c(s) = c_{\text{puct}} + \log\left(\frac{N(s) + c_{\text{base}} + 1}{c_{\text{base}}}\right)$

When a leaf node $s_L$ is reached, it is expanded and evaluated using the value head $v = V_\theta(s_L)$. The value $v$ is backpropagated to update all ancestors:
$Q(s, a) \leftarrow Q(s, a) + \frac{v - Q(s, a)}{N(s, a)}$
$N(s, a) \leftarrow N(s, a) + 1$

### Asymptotic Convergence Analysis
Under the PUCT selection rule, the search tree converges to the optimal policy. 

#### Theorem 3: Asymptotic Consistency of PUCT
Assume that the state space is finite and the action values are bounded. As the number of search simulations $M \to \infty$:
1. The estimated action-values converge to the true optimal action-values:
   $\lim_{M \to \infty} Q(s, a) = q^*(s, a) \quad \text{almost surely}$
2. The visit distribution converges to the optimal policy:
   $\lim_{M \to \infty} \frac{N(s, a)}{N(s)} = \pi^*(a|s)$

*Proof Sketch:*
The exploration bonus term $U(s, a)$ is proportional to $\frac{\sqrt{N(s)}}{1 + N(s, a)}$. If an action $a$ is sub-optimal (i.e., $q^*(s, a) < \max_{a'} q^*(s, a')$), its selection frequency will be constrained. If $N(s, a)$ were to remain finite while $N(s) \to \infty$, the exploration term $U(s, a) \propto \sqrt{N(s)}$ would grow unbounded and eventually exceed the value difference. This forces the search to explore all actions infinitely often:
$\lim_{N(s) \to \infty} N(s, a) = \infty \quad \forall a \in \mathcal{A}$
Since all descendant subtrees are visited infinitely often, the recursive value estimates backpropagated from the leaf nodes converge to the true values by the law of large numbers. Because the mean value $Q(s, a) \to q^*(s, a)$ and the exploration term $U(s, a) \to 0$ relative to $Q(s, a)$, the selection rule asymptotically prioritizes the actions maximizing $q^*(s, a)$, guaranteeing that $\frac{N(s, a)}{N(s)}$ converges to the optimal policy $\pi^*(a|s)$. $\blacksquare$

---

## 3. Heuristic-Guided Policy Gradient

To bypass the inefficiency of random exploration in early-stage reinforcement learning, we can mathematically steer the policy optimization using an auxiliary loss function based on the **Kullback-Leibler (KL) Divergence** against a baseline heuristic policy $P_H(a|s)$.

### Loss Function
Let $\pi_\theta(a|s)$ be the policy parameterized by the network weights $\theta$. The total loss function $L(\theta)$ is defined as:
$L(\theta) = L_{PG}(\theta) + \beta \cdot D_{KL}(P_H(s) \parallel \pi_\theta(s)) + \frac{1}{2} L_V(\theta)$
where:
* **Policy Gradient Loss**:
  $L_{PG}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \log \pi_\theta(a_i | s_i) A_i$
  with advantages $A_i = G_i - V_\theta(s_i)$.
* **KL Divergence Regularization**:
  $D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \sum_{a \in \mathcal{A}} P_H(a|s) \log \left( \frac{P_H(a|s)}{\pi_\theta(a|s)} \right)$
* **Value Head MSE Loss**:
  $L_V(\theta) = \frac{1}{|B|} \sum_{i=1}^{|B|} (G_i - V_\theta(s_i))^2$

### Theoretical Derivation of the Loss Gradient
We derive the gradient of the total policy objective with respect to the network parameters $\theta$:

$\nabla_\theta L_{\text{policy}}(\theta) = \nabla_\theta \left( L_{PG}(\theta) + \beta \cdot D_{KL}(P_H(s) \parallel \pi_\theta(s)) \right)$

First, expanding the KL divergence term:
$D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \sum_{a \in \mathcal{A}} P_H(a|s) \log P_H(a|s) - \sum_{a \in \mathcal{A}} P_H(a|s) \log \pi_\theta(a|s)$

Taking the gradient with respect to $\theta$:
$\nabla_\theta D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \nabla_\theta \left( \sum_{a \in \mathcal{A}} P_H(a|s) \log P_H(a|s) \right) - \nabla_\theta \left( \sum_{a \in \mathcal{A}} P_H(a|s) \log \pi_\theta(a|s) \right)$

Since the heuristic policy $P_H(a|s)$ is independent of the model parameters $\theta$, the gradient of the first term is zero:
$\nabla_\theta D_{KL}(P_H(s) \parallel \pi_\theta(s)) = - \sum_{a \in \mathcal{A}} P_H(a|s) \nabla_\theta \log \pi_\theta(a|s)$

Next, taking the gradient of the empirical Policy Gradient loss:
$\nabla_\theta L_{PG}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \nabla_\theta \log \pi_\theta(a_i | s_i) A_i = - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_i = a) A_i \nabla_\theta \log \pi_\theta(a | s_i)$
where $\mathbb{I}$ is the indicator function.

Combining the gradients of both components yields:
$\nabla_\theta L_{\text{policy}}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_i = a) A_i \nabla_\theta \log \pi_\theta(a | s_i) - \beta \cdot \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} P_H(a | s_i) \nabla_\theta \log \pi_\theta(a | s_i)$

Factoring out the shared terms:
$\nabla_\theta L_{\text{policy}}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \left[ \mathbb{I}(a_i = a) A_i + \beta P_H(a | s_i) \right] \nabla_\theta \log \pi_\theta(a | s_i) \quad \blacksquare$

#### Interpretation
The term $\beta P_H(a | s_i)$ mathematically functions as a **pseudo-advantage**. In early stages of training (where $\beta$ is large), the updates are heavily guided to mimic the heuristic, preventing erratic random movements. As training progresses, $\beta$ is decayed geometrically:
$\beta_{t+1} = \max(\beta_t \cdot \gamma_{decay}, \beta_{min})$
which gradually shifts the optimization priority entirely toward the actual RL return advantages $A_i$.

### PPO Clipped Objective & Gradient Derivation
When using Proximal Policy Optimization (PPO), the standard policy gradient is replaced by a clipped surrogate objective to prevent large policy updates. Let $r_i(\theta) = \frac{\pi_\theta(a_i | s_i)}{\pi_{\theta_{\text{old}}}(a_i | s_i)}$ be the probability ratio. The clipped objective is defined as:
$L_{CLIP}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \min \left( r_i(\theta) A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right)$
where $\epsilon$ is the clipping hyperparameter (typically $0.2$).

#### Theorem 4: Gradient of the PPO Clipped Loss with Heuristic Guidance
The gradient of the hybrid PPO clipped loss $L_{\text{policy}}^{\text{PPO}}(\theta) = L_{CLIP}(\theta) + \beta D_{KL}(P_H(s) \parallel \pi_\theta(s))$ with respect to $\theta$ is:
$\nabla_\theta L_{\text{policy}}^{\text{PPO}}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \left[ \mathbb{I}(a_i = a) M_i(\theta) r_i(\theta) A_i + \beta P_H(a | s_i) \right] \nabla_\theta \log \pi_\theta(a | s_i)$
where $M_i(\theta)$ is a clipping indicator function:
$M_i(\theta) = \begin{cases}
1 & \text{if } r_i(\theta) A_i < \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \\
0 & \text{otherwise}
\end{cases}$

*Proof:*
We first calculate the gradient of the probability ratio $r_i(\theta)$:
$\nabla_\theta r_i(\theta) = \frac{\nabla_\theta \pi_\theta(a_i | s_i)}{\pi_{\theta_{\text{old}}}(a_i | s_i)} = \frac{\pi_\theta(a_i | s_i) \nabla_\theta \log \pi_\theta(a_i | s_i)}{\pi_{\theta_{\text{old}}}(a_i | s_i)} = r_i(\theta) \nabla_\theta \log \pi_\theta(a_i | s_i)$

Under the minimum operator, the gradient of the term $u_i(\theta) = \min \left( r_i(\theta) A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right)$ is non-zero only when the unclipped value is smaller than the clipped value, which is denoted by the indicator $M_i(\theta)$:
$\nabla_\theta u_i(\theta) = M_i(\theta) \nabla_\theta r_i(\theta) A_i = M_i(\theta) r_i(\theta) A_i \nabla_\theta \log \pi_\theta(a_i | s_i)$

Rewriting using the summation over all actions $a \in \mathcal{A}$ with the coordinate indicator $\mathbb{I}(a_i = a)$:
$\nabla_\theta L_{CLIP}(\theta) = - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_i = a) M_i(\theta) r_i(\theta) A_i \nabla_\theta \log \pi_\theta(a | s_i)$

Adding the gradient of the KL Divergence term derived in the previous section:
$\nabla_\theta L_{\text{policy}}^{\text{PPO}}(\theta) = \nabla_\theta L_{CLIP}(\theta) + \beta \nabla_\theta D_{KL}(P_H(s) \parallel \pi_\theta(s))$
$= - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_i = a) M_i(\theta) r_i(\theta) A_i \nabla_\theta \log \pi_\theta(a | s_i) - \beta \cdot \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} P_H(a | s_i) \nabla_\theta \log \pi_\theta(a | s_i)$
$= - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{a \in \mathcal{A}} \left[ \mathbb{I}(a_i = a) M_i(\theta) r_i(\theta) A_i + \beta P_H(a | s_i) \right] \nabla_\theta \log \pi_\theta(a | s_i) \quad \blacksquare$

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
