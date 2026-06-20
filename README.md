# Advanced Grid-based Reinforcement Learning Theories & Implementations

A research-oriented repository detailing three mathematical formulations and complete PyTorch implementations for state-of-the-art grid-based reinforcement learning (RL):
1. **Equivariant Neural Networks (Dihedral $D_4$ Symmetry)**
2. **Actor-Critic MCTS Hybrid Search**
3. **Heuristic-Guided Policy Gradient via KL Divergence**

---

## Table of Contents
1. [English Guide](#english-guide)
   - [1. Equivariant Neural Networks](#1-equivariant-neural-networks)
   - [2. Actor-Critic MCTS Integration](#2-actor-critic-mcts-integration)
   - [3. Heuristic-Guided Policy Gradient](#3-heuristic-guided-policy-gradient)
2. [Korean Guide (한글 가이드)](#korean-guide-한글-가이드)
   - [1. 동변 신경망 (Equivariant Neural Networks)](#1-동변-신경망-equivariant-neural-networks)
   - [2. 액터-크리틱 MCTS 결합](#2-액터-크리틱-mcts-결합)
   - [3. 휴리스틱 가이드 정책 그래디언트](#3-휴리스틱-가이드-정책-그래디언트)

---

# English Guide

## 1. Equivariant Neural Networks
In 2D grid-based board games (e.g., Gomoku, Chess, Go, Tic-Tac-Toe), the board possesses physical symmetries under rotation and reflection. The group representing these operations on a square grid is the **Dihedral Group $D_4$**, which contains 8 group elements (4 rotations and 4 reflections).

### Mathematical Definition
A neural network function $f: X \to Y$ is said to be **equivariant** with respect to a symmetry group $G$ if transforming the input $x$ by a group element $g \in G$ yields the same result as transforming the output $f(x)$ by $g$:

$$f(g \cdot x) = g \cdot f(x) \quad \forall g \in G$$

For a policy network $\pi_\theta(a|s)$ predicting move probabilities over a grid:
$$\pi_\theta(g \cdot a \mid g \cdot s) = g \cdot \pi_\theta(a \mid s)$$

By enforcing equivariance directly into the neural network architecture, we constrain the hypothesis space to physically valid symmetrical functions. 
- **Benefit**: This mathematically guarantees that the agent treats symmetric states identically, reducing the search space and eliminating the need for 8x data augmentation, leading to **up to 8x faster training convergence**.

---

## 2. Actor-Critic MCTS Integration
Traditional Monte Carlo Tree Search (MCTS) relies on random playouts (rollouts) to evaluate leaf nodes. However, random rollouts exhibit high variance and consume massive computational time. In AlphaZero-like architectures, a dual-head network replaces rollouts with deep value evaluations.

### Formulation
A single neural network takes the board state $s$ as input and produces two outputs:
$$(\mathbf{p}, v) = f_\theta(s)$$
Where:
- $\mathbf{p} \in \mathbb{R}^{|A|}$ is the policy vector (prior probabilities $\pi(a|s)$).
- $v \in [-1, 1]$ is the scalar state value estimating the expected game outcome (Value).

### Search & Update Logic
During the selection phase, MCTS chooses actions maximizing the Upper Confidence Bound (UCB) variant (PUCT):

$$a_t = \arg\max_a \left( Q(s, a) + U(s, a) \right)$$

$$U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

Where $P(s, a)$ is directly populated by the network's policy head $\mathbf{p}$. When a leaf node $s_L$ is expanded, instead of simulating to the end of the game, it is immediately evaluated using the value head:
$$v = V_\theta(s_L)$$
The value $v$ is then backpropagated up the tree to update the action-value estimate $Q(s, a)$.

---

## 3. Heuristic-Guided Policy Gradient
Standard reinforcement learning starts with random exploration, which is highly inefficient for games with vast search spaces. We can accelerate early-stage training by steering the agent's policy toward established human heuristics using an auxiliary loss function based on **Kullback-Leibler (KL) Divergence**.

### Theoretical Derivation
Let $\pi_\theta(a|s)$ be the trainable neural network policy, and $P_H(a|s)$ be a probability distribution derived from a heuristic rule (e.g., scoring lines, threats, or blocking patterns).

The difference between the two distributions is measured using the KL Divergence:

$$D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \sum_{a \in A} P_H(a|s) \log \left( \frac{P_H(a|s)}{\pi_\theta(a|s)} \right)$$

We define the total loss function by combining the standard Policy Gradient loss $L_{PG}(\theta)$ and the heuristic regularization loss:

$$L(\theta) = L_{PG}(\theta) + \beta \cdot D_{KL}(P_H(s) \parallel \pi_\theta(s))$$

Where:
- $L_{PG}(\theta) = - \log \pi_\theta(a|s) \cdot G_t$
- $\beta \ge 0$ is a dynamic weight coefficient that decays as training progresses (e.g., $\beta_t = \beta_0 \cdot \gamma^t$).
- **Benefit**: In the initial stages, the agent is guided by the heuristic, preventing useless random movements. As $\beta$ decays to $0$, the agent transitions into pure self-play RL, enabling it to surpass the limits of the heuristic.

---

# Korean Guide (한글 가이드)

## 1. 동변 신경망 (Equivariant Neural Networks)
오목, 체스, 바둑 등 2D 격자판 기반 보드게임은 회전이나 대칭 변환에 대해 물리적 대칭성을 가집니다. 정사각형 격자에서 이러한 연산을 나타내는 수학적 대칭군은 **이면군 $D_4$ (Dihedral Group $D_4$)**이며, 총 8개의 원소(회전 4개, 반사 4개)로 구성됩니다.

### 수학적 정의
대칭군 $G$의 원소 $g \in G$에 대해, 입력 $x$를 변환시킨 결과에 신경망 $f$를 적용한 값이 신경망의 출력 $f(x)$를 변환시킨 결과와 동일할 때, 함수 $f$는 **동변(Equivariant)**하다고 정의합니다:

$$f(g \cdot x) = g \cdot f(x) \quad \forall g \in G$$

이를 격자판 위 착수 확률을 출력하는 정책 신경망 $\pi_\theta(a|s)$에 적용하면 다음과 같습니다:
$$\pi_\theta(g \cdot a \mid g \cdot s) = g \cdot \pi_\theta(a \mid s)$$

이러한 대칭 규칙을 신경망 아키텍처 자체에 제약 조건으로 주입함으로써 학습 범위를 물리적으로 유효한 대칭 함수 공간으로 압축합니다.
- **장점**: 대칭 상태를 완전히 동일하게 평가함을 수학적으로 보장하므로, 8배의 데이터 증강이 필요 없고 **학습 수렴 속도가 최대 8배까지 빨라집니다**.

---

## 2. 액터-크리틱 MCTS 결합
기존의 몬테카를로 트리 탐색(MCTS)은 리프 노드를 평가하기 위해 끝까지 무작위 대국(롤아웃)을 수행해야 했습니다. 그러나 이는 연산량이 막대하고 확률적 분산이 큽니다. AlphaZero 스타일의 아키텍처에서는 이를 정책-가치 듀얼 헤드 네트워크로 대체합니다.

### 공식화
신경망은 보드 상태 $s$를 입력받아 두 가지 핵심 값을 동시에 도출합니다:
$$(\mathbf{p}, v) = f_\theta(s)$$
- $\mathbf{p} \in \mathbb{R}^{|A|}$: 착수 후보들에 대한 사전 확률 분포 (Policy $\pi(a|s)$).
- $v \in [-1, 1]$: 현재 형세를 평가한 예상 승률 수치 (Value).

### 탐색 및 업데이트
트리 탐색 시 사전 확률과 탐색 횟수를 고려하여 UCB(Upper Confidence Bound)를 극대화하는 행동을 선택합니다:

$$a_t = \arg\max_a \left( Q(s, a) + U(s, a) \right)$$

$$U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

여기서 사전 확률 $P(s, a)$에 신경망이 예측한 정책 $\mathbf{p}$의 값이 주입됩니다. 리프 노드 $s_L$에 도달했을 때 롤아웃 시뮬레이션을 수행하는 대신, 밸류 헤드를 거쳐 즉시 상태 가치를 평가합니다:
$$v = V_\theta(s_L)$$
이 평가값 $v$를 트리 경로 상부로 역전파(Backpropagation)하여 $Q(s, a)$ 값을 업데이트합니다.

---

## 3. 휴리스틱 가이드 정책 그래디언트
강화학습 초기의 무작위 탐색은 광활한 수읽기 영역에서 매우 비효율적입니다. **쿨백-라이블러 발산(Kullback-Leibler Divergence)**을 사용한 보조 손실 함수를 설계하여, 학습 초기 단계에 인간이 정립한 룰(휴리스틱 함수 $P_H$)의 안내를 받게 함으로써 수렴을 극대화할 수 있습니다.

### 이론 유도
학습할 신경망 정책을 $\pi_\theta(a|s)$, 오목 룰이나 점수표 기반의 휴리스틱 정책을 $P_H(a|s)$라고 정의합니다.

두 확률 분포의 거리적 차이는 다음과 같이 KL Divergence 수식으로 측정됩니다:

$$D_{KL}(P_H(s) \parallel \pi_\theta(s)) = \sum_{a \in A} P_H(a|s) \log \left( \frac{P_H(a|s)}{\pi_\theta(a|s)} \right)$$

전체 학습 손실 함수(Loss Function)는 표준 정책 그래디언트 손실 $L_{PG}(\theta)$에 이 가이드 손실을 규제 항목으로 합성하여 정의합니다:

$$L(\theta) = L_{PG}(\theta) + \beta \cdot D_{KL}(P_H(s) \parallel \pi_\theta(s))$$

- $L_{PG}(\theta) = - \log \pi_\theta(a|s) \cdot G_t$
- $\beta \ge 0$는 학습 진행도에 따라 서서히 감소(Decay)하는 가중치 계수입니다 (예: $\beta_t = \beta_0 \cdot \gamma^t$).
- **장점**: 학습 초반에는 강력한 휴리스틱 가이드에 의해 탐색 공간이 올바른 방향으로 규제되지만, 학습 후반에 $\beta$가 0에 수렴하면서 순수한 자가대국 강화학습으로 전향해 휴리스틱 자체의 한계를 뛰어넘을 수 있게 됩니다.

---

## Citation & Intellectual Property
If you use this work, theoretical formulations, or implementation code in your research or projects, please cite it as follows:

```bibtex
@misc{wonc_equivariant_guided_rl_2026,
  author       = {WonC-Lab},
  title        = {Advanced Grid-Based Reinforcement Learning Theories with Equivariant CNN and Heuristic-Guided Loss},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/WonC-Lab/WonC-Equivariant-Guided-RL}}
}
```

### License
This repository and all its theoretical derivations, mathematical formulations, and implementation codes are owned by **WonC-Lab**. They are licensed under the **MIT License**.
Copyright (c) 2026 WonC-Lab. All rights reserved.

---

## 인용 및 지적 재산권
본 연구 성과, 수학적 이론 유도 공식 및 소스코드를 학술 연구나 개인 프로젝트에 인용할 경우 아래 서식을 사용해 주시기 바랍니다:

```bibtex
@misc{wonc_equivariant_guided_rl_2026,
  author       = {WonC-Lab},
  title        = {Advanced Grid-Based Reinforcement Learning Theories with Equivariant CNN and Heuristic-Guided Loss},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/WonC-Lab/WonC-Equivariant-Guided-RL}}
}
```

### 라이선스
본 리포지토리의 모든 이론 공식과 소스코드의 소유권은 **WonC-Lab**에 있으며, **MIT 라이선스** 하에 제공됩니다.
Copyright (c) 2026 WonC-Lab. All rights reserved.
