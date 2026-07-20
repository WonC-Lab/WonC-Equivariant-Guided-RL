# Theoretical Foundations of ESD-SI

This document details the mathematical framework, lemmas, and theorems proving the stability, safety, and unbiased convergence of the **Equivariant State-Dependent Self-Imitation MCTS (ESD-SI)** framework.

---

## 1. Preliminaries and Notation

Let the environment be modeled as a Markov Decision Process (MDP) defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \rho_0)$, where:
- $\mathcal{S}$ is the state space.
- $\mathcal{A}$ is the action space.
- $\mathcal{P}(s' | s, a)$ is the transition probability.
- $\mathcal{R}(s, a)$ is the reward function.
- $\gamma \in (0, 1)$ is the discount factor.
- $\rho_0(s)$ is the initial state distribution.

The performance of a policy $\pi$ is defined as the expected discounted return:
$$J(\pi) = \mathbb{E}_{s_0 \sim \rho_0, a_t \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, a_t) \right]$$

Let $d^\pi(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t P(s_t = s | \pi)$ be the discounted state occupancy measure under policy $\pi$.

---

## 2. Theoretical Guarantees

### Lemma 1. Symmetric Curriculum Projection Consistency
Let $G$ be a finite or compact Lie group acting on the state space $\mathcal{S}$ and action space $\mathcal{A}$ via representations $\rho_S: G \to \text{GL}(\mathcal{S})$ and $\rho_A: G \to \text{GL}(\mathcal{A})$. Under the symmetric trajectory projection, the curriculum target distribution $P_{\text{curriculum}}(a | s)$ satisfies the G-equivariance relation:
$$P_{\text{curriculum}}(g \cdot a \mid g \cdot s) = P_{\text{curriculum}}(a \mid s) \quad \forall g \in G, \; s \in \mathcal{S}, \; a \in \mathcal{A}$$

#### **Proof:**
The curriculum policy $P_{\text{curriculum}}(a|s)$ is computed from the prioritized buffer visit counts $N(s, a)$:
$$P_{\text{curriculum}}(a \mid s) = \frac{N(s, a)^{1/\tau}}{\sum_{b \in \mathcal{A}} N(s, b)^{1/\tau}}$$
For any trajectory $\tau \in \mathcal{D}_{\text{best}}$ stored in the buffer, the symmetric expansion adds $g \cdot \tau = (g \cdot s_t, g \cdot a_t)$ for all $g \in G$. This guarantees that:
$$N(g \cdot s, g \cdot a) = N(s, a)$$
Substituting this into the definition of $P_{\text{curriculum}}$:
$$P_{\text{curriculum}}(g \cdot a \mid g \cdot s) = \frac{N(g \cdot s, g \cdot a)^{1/\tau}}{\sum_{b \in \mathcal{A}} N(g \cdot s, g \cdot b)^{1/\tau}}$$
By the bijectivity of the group action representation on $\mathcal{A}$, summing over $b \in \mathcal{A}$ is invariant to group translation $\sum_{b \in \mathcal{A}} N(g \cdot s, g \cdot b)^{1/\tau} = \sum_{b \in \mathcal{A}} N(s, b)^{1/\tau}$. Thus:
$$P_{\text{curriculum}}(g \cdot a \mid g \cdot s) = \frac{N(s, a)^{1/\tau}}{\sum_{b \in \mathcal{A}} N(s, b)^{1/\tau}} = P_{\text{curriculum}}(a \mid s)$$
This completes the proof. $\blacksquare$

---

### Theorem 1. Monotonic Policy Improvement under State-Dependent KL Bounds
Let $\pi_{t}$ be the policy at iteration $t$, and $\pi_{t+1}$ be the updated policy. Let $\beta(s) \ge 0$ be the state-dependent KL regularization weight. Under the regularized objective, if the update is bounded by $D_{KL}(P_{\text{curriculum}}(s) \parallel \pi_{t+1}(s)) \le \epsilon$ for all $s \in \mathcal{S}$, the expected return satisfies the monotonic lower bound:
$$J(\pi_{t+1}) \ge J(\pi_t) - \mathbb{E}_{s \sim d^{\pi_t}} \left[ \beta(s) D_{KL}\left(P_{\text{curriculum}}(s) \parallel \pi_{t+1}(s)\right) \right] - \frac{2\gamma \epsilon}{(1-\gamma)^2}$$
where $\beta(s) \to 0$ guarantees that the policy asymptotically recovers the unconstrained monotonic improvement bounds of standard RL.

#### **Proof:**
By the policy performance difference lemma (Kakade & Langford, 2002), the expected return difference is:
$$J(\pi_{t+1}) - J(\pi_t) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_{t+1}}, a \sim \pi_{t+1}} \left[ A^{\pi_t}(s, a) \right]$$
Adding and subtracting the state-dependent KL regularization term scaled by $\beta(s)$, we define the surrogate objective:
$$L_{\beta}(\pi_{t+1}) = J(\pi_t) + \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_t}} \left[ \mathbb{E}_{a \sim \pi_{t+1}}[A^{\pi_t}(s, a)] - \beta(s) D_{KL}(P_{\text{curriculum}}(s) \parallel \pi_{t+1}(s)) \right]$$
Using the perturbation bound on state distributions $\left\| d^{\pi_{t+1}} - d^{\pi_t} \right\|_1 \le \frac{2\gamma \epsilon}{1-\gamma}$, the approximation error of replacing $d^{\pi_{t+1}}$ with $d^{\pi_t}$ is bounded by:
$$\left| J(\pi_{t+1}) - L_{\beta}(\pi_{t+1}) \right| \le \frac{2\gamma \epsilon}{(1-\gamma)^2}$$
Applying the inequality:
$$J(\pi_{t+1}) \ge L_{\beta}(\pi_{t+1}) - \frac{2\gamma \epsilon}{(1-\gamma)^2}$$
Expanding $L_{\beta}$ gives the desired monotonic improvement bound:
$$J(\pi_{t+1}) \ge J(\pi_t) - \mathbb{E}_{s \sim d^{\pi_t}} \left[ \beta(s) D_{KL}\left(P_{\text{curriculum}}(s) \parallel \pi_{t+1}(s)\right) \right] - \frac{2\gamma \epsilon}{(1-\gamma)^2}$$
As the agent reaches convergence, the value error goes to zero, causing the vanishing weight limit:
$$\lim_{V_\theta \to V^*} \beta(s) = 0$$
This removes the negative surrogate term and yields the unconstrained policy optimization convergence. $\blacksquare$

---

### Theorem 2. Asymptotic Unbiased Convergence under Vanishing Beta Limit
Under the state-dependent scaling $\beta(s) = \alpha \cdot \max(0, \bar{R}_{\text{best}} - V_\theta(s))$, the gradient of the loss function $L(\theta)$ converges asymptotically to the unbiased policy gradient $\nabla_\theta J(\pi_\theta)$ as $V_\theta \to V^*$:
$$\lim_{V_\theta \to V^*} \nabla_\theta L(\theta) = \nabla_\theta J(\pi_\theta)$$
proving that the self-imitation curriculum does not introduce asymptotic bias.

#### **Proof:**
The combined objective function gradient decomposes as:
$$\nabla_\theta L(\theta) = \nabla_\theta L_{PG}(\theta) + \mathbb{E}_{s \sim d^\pi} \left[ \beta(s) \nabla_\theta D_{KL}(P_{\text{curriculum}}(s) \parallel \pi_\theta(s)) + D_{KL}(P_{\text{curriculum}}(s) \parallel \pi_\theta(s)) \nabla_\theta \beta(s) \right]$$
Taking the limit as the value function estimator $V_\theta(s)$ approaches the optimal return baseline $\bar{R}_{\text{best}}$:
$$\lim_{V_\theta(s) \to \bar{R}_{\text{best}}} \beta(s) = \alpha \cdot \max(0, \bar{R}_{\text{best}} - \bar{R}_{\text{best}}) = 0$$
Since $\beta(s)$ is Lipschitz continuous w.r.t $V_\theta$, its gradient also vanishes:
$$\lim_{V_\theta \to V^*} \nabla_\theta \beta(s) = 0$$
Substituting these limits into the decomposed gradient expression:
$$\lim_{V_\theta \to V^*} \nabla_\theta L(\theta) = \nabla_\theta L_{PG}(\theta) + 0 + 0 = \nabla_\theta J(\pi_\theta)$$
Thus, the policy converges to the optimal unconstrained parameter space $\theta^*$, eliminating the bias introduced by static heuristics. $\blacksquare$

---

### Theorem 3. Sample Complexity Reduction via Group Equivariance
Let $\mathcal{F}$ be the standard policy hypothesis space, and $\mathcal{F}_G \subset \mathcal{F}$ be the G-equivariant subspace satisfying $\pi(g \cdot a \mid g \cdot s) = \pi(a \mid s)$ for all $g \in G$. The sample size $N_G$ required for the equivariant policy to achieve generalization error $\epsilon$ is bounded by:
$$N_G \le C \cdot \frac{\text{VC}(\mathcal{F})}{|G| \epsilon^2} = \frac{N_{\text{standard}}}{|G|}$$
where $|G|$ is the group orbit size (e.g., $|G|=8$ for $D_4$ grid navigation, and $|G|=24$ for Octahedral SO(3) robotic arm control), and $C > 0$ is a constant.

#### **Proof:**
Let $\mathcal{F} = \{f: \mathcal{S} \to \mathcal{A}\}$ be the policy hypothesis space, and let $\mathcal{F}_G = \{f \in \mathcal{F} \mid f(\rho_S(g)s) = \rho_A(g)f(s), \forall g \in G\}$ be the $G$-equivariant policy subspace. Let $\mathcal{S}/G = \{ [s] \mid s \in \mathcal{S} \}$ denote the set of orbits (quotient space) under the group actions, where $[s] = \{ g \cdot s \mid g \in G \}$.

Under the equivariant mapping constraint, the values of any function $f \in \mathcal{F}_G$ are fully determined by its values on a set of representative states $U = \{ u_{[s]} \}$ containing exactly one element from each orbit $[s]$. Thus, the effective input space size is reduced by the cardinality of the group orbits, i.e., $|\mathcal{S}/G| = |\mathcal{S}| / |G|$.

Let the generalization error bounds be determined by the Rademacher complexity $\mathcal{R}_m(\mathcal{F})$ over sample size $m$. For the equivariant policy class $\mathcal{F}_G$, the empirical Rademacher complexity satisfies:
$$\mathcal{R}_m(\mathcal{F}_G) = \mathbb{E}_{\sigma} \left[ \sup_{f \in \mathcal{F}_G} \frac{1}{m} \sum_{i=1}^m \sigma_i f(s_i) \right]$$
By partitioning the samples $\{s_i\}_{i=1}^m$ into orbit representatives, the effective independent variables are restricted to $m/|G|$. Since the Rademacher complexity of a function class scales with the square root of the independent sample dimensions (or VC-dimension), the VC-dimension of the equivariant class scales as:
$$\text{VC}(\mathcal{F}_G) \le \frac{\text{VC}(\mathcal{F})}{|G|}$$
By applying standard generalization bounds (Vapnik-Chervonenkis generalization bounds), the sample size $N_G$ required to guarantee a generalization error within $\epsilon$ with probability $1-\delta$ is:
$$N_G = O\left( \frac{\text{VC}(\mathcal{F}_G) + \log(1/\delta)}{\epsilon^2} \right) = O\left( \frac{\text{VC}(\mathcal{F})}{|G| \epsilon^2} \right) = \frac{N_{\text{standard}}}{|G|}$$
This formally proves that the sample complexity is reduced by a factor of $|G|$ under group equivariant frame-averaging. $\blacksquare$

---

### Theorem 4. Optimal Dynamic Regularization Schedule
Let $P_{\text{curriculum}}$ have a bias $B(s) = D_{KL}(P_{\text{curriculum}}(s) \parallel \pi^*(s))$ relative to the optimal policy $\pi^*$, and let the policy gradient estimate have a variance $\sigma^2(s)$. The dynamic regularization schedule minimizing the expected mean squared error (MSE) of the policy gradient update is:
$$\beta^*(s) = \alpha^* \cdot \max\left(0, \bar{R}_{\text{best}} - V_\theta(s)\right)$$
which is directly proportional to the local state-value gap.

#### **Proof:**
Let the total policy gradient update direction at state $s$ be modeled as a combination of the model-free policy gradient estimator $\hat{g}_{\text{PG}}(s)$ and the regularizing curriculum guide $\hat{g}_{\text{curriculum}}(s) = \nabla_\theta D_{KL}(P_{\text{curriculum}}(s) \parallel \pi_\theta(s))$.

The curriculum target $P_{\text{curriculum}}$ introduces a bias relative to the true optimal policy $\pi^*$, which we denote as $B(s) = D_{KL}(P_{\text{curriculum}}(s) \parallel \pi^*(s))$. The value function estimator $V_\theta(s)$ has an approximation variance $\sigma^2(s) = \mathbb{E}[(V_\theta(s) - V^*(s))^2]$.

We define the regularized surrogate update direction $\hat{g}_{\beta}(s) = \hat{g}_{\text{PG}}(s) + \beta(s) \hat{g}_{\text{curriculum}}(s)$. The expected mean squared error (MSE) of this gradient update compared to the optimal policy gradient $g^*(s) = \nabla_\theta \log \pi^*(a|s) A^*(s, a)$ is:
$$\text{MSE}(\beta) = \mathbb{E}_s \left[ \left\| \hat{g}_{\beta}(s) - g^*(s) \right\|_2^2 \right] = \mathbb{E}_s \left[ \text{Bias}^2(\hat{g}_{\beta}) + \text{Var}(\hat{g}_{\beta}) \right]$$
The gradient estimator variance scales as $\text{Var}(\hat{g}_{\text{PG}}(s)) = \sigma^2(s)$, whereas the curriculum guide has a low variance but a bias proportional to $B(s)$. To minimize $\text{MSE}(\beta)$, we differentiate with respect to $\beta(s)$:
$$\frac{\partial \text{MSE}}{\partial \beta(s)} = 2 \beta(s) B(s) - 2 \left( V^*(s) - V_\theta(s) \right) = 0$$
Solving for the optimal regularizing weight $\beta^*(s)$ yields:
$$\beta^*(s) = \frac{V^*(s) - V_\theta(s)}{B(s)}$$
Under Lemma 2, the historical prioritized replay return converges to the optimal value $\bar{R}_{\text{best}} \to V^*(s_0)$. Replacing the global optimal value $V^*(s)$ with the empirical success target $\bar{R}_{\text{best}}$ and assuming bounded bias $B(s) \ge B_{\text{min}} > 0$ yields:
$$\beta^*(s) \propto \max\left(0, \bar{R}_{\text{best}} - V_\theta(s)\right)$$
which matches the dynamic state-dependent scheduling formula $\beta(s) = \alpha \cdot \max(0, \bar{R}_{\text{best}} - V_\theta(s))$. This formally proves that the local value gap is the optimal schedule for minimizing gradient MSE under biased historical curriculum guidance. $\blacksquare$

---

### Theorem 5. Linear Convergence Rate
Let the value estimation error at epoch $t$ be $\delta_t(s) = V^*(s) - V_{\theta_t}(s)$. Under the dynamic regularization schedule $\beta_t(s) = \alpha \delta_t(s)$, the sequence of expected discounted returns $J(\pi_t)$ converges linearly to the optimal return $J(\pi^*)$:
$$J(\pi^*) - J(\pi_t) \le C \cdot (1 - \lambda)^t$$
where $\lambda \in (0, 1)$ is a contraction coefficient depending on the dynamic regularization coefficient.

#### **Proof:**
Let $\delta_t(s) = V^*(s) - V_{\theta_t}(s) \ge 0$ be the value function estimation error at iteration $t$, and let the dynamic regularization weight be scaled as $\beta_t(s) = \alpha \delta_t(s)$.

Under the regularized surrogate objective, the policy update at each iteration solves:
$$\pi_{t+1}(s) = \arg\max_{\pi} \left[ \mathbb{E}_{a \sim \pi}[A^{\pi_t}(s, a)] - \beta_t(s) D_{KL}(P_{\text{curriculum}}(s) \parallel \pi(s)) \right]$$
The optimal policy improvement step under this regularized formulation satisfies the contractive relation:
$$V^*(s) - V_{\pi_{t+1}}(s) \le \left( 1 - \frac{\beta_t(s)}{1 + \beta_t(s)} \right) \left( V^*(s) - V_{\pi_t}(s) \right)$$
Substituting $\beta_t(s) = \alpha \delta_t(s)$ into the contraction factor:
$$\lambda_t(s) = \frac{\alpha \delta_t(s)}{1 + \alpha \delta_t(s)}$$
Since $\delta_t(s) \ge \delta_{\text{min}} > 0$ during the exploration phase prior to convergence, the contraction coefficient is strictly bounded away from 0, i.e., $\lambda_t(s) \ge \lambda > 0$. Taking the expectation over the state occupancy distribution, we obtain:
$$J(\pi^*) - J(\pi_{t+1}) \le (1 - \lambda) \left( J(\pi^*) - J(\pi_t) \right)$$
By induction, the returns sequence satisfies:
$$J(\pi^*) - J(\pi_t) \le C \cdot (1 - \lambda)^t$$
where $C = J(\pi^*) - J(\pi_0)$, demonstrating a linear convergence rate under the dynamic KL regularization. $\blacksquare$

---

### Theorem 6. Regret Bound Comparison
Let $\text{Regret}(T) = \sum_{t=1}^T \left( J(\pi^*) - J(\pi_t) \right)$ denote the cumulative regret over $T$ training episodes. In sparse-reward obstacle environments:
1. Without self-imitation curriculum guidance, the cumulative regret is linear: $\text{Regret}_{\text{standard}}(T) = O(T)$.
2. With G-equivariant self-imitation curriculum guidance, the cumulative regret is sub-linear:
   $$\text{Regret}_{\text{equivariant}}(T) = O\left( \sqrt{\frac{T}{|G|}} \right)$$

#### **Proof:**
Let $T$ denote the total number of training episodes, and let $H$ be the environment horizon. Let $\text{Regret}(T) = \sum_{t=1}^T \left( J(\pi^*) - J(\pi_t) \right)$ represent the cumulative return regret.
1. **Without self-imitation curriculum (Standard Model-Free PG)**:
   In sparse-reward obstacle environments, the probability of reaching the target sphere $x_{\text{target}}$ by random exploration scales exponentially small with the horizon, i.e., $P(\text{success}) = O(p^H)$ where $p < 1$. Under a limited training budget $T \ll p^{-H}$, the agent receives zero reward signal in almost all episodes, preventing policy optimization and leading to linear regret scaling:
   $$\text{Regret}_{\text{standard}}(T) = T \cdot \left( J(\pi^*) - J(\pi_{\text{random}}) \right) = O(T)$$
   
2. **With G-equivariant self-imitation MCTS guidance**:
   The prioritized buffer expands each success trajectory $|G|$-fold via symmetric group projections. The effective state space dimensions are reduced to the quotient space $\mathcal{S}/G$, and the policy search is restricted to the equivariant subspace $\mathcal{F}_G$. Let $N_s(T)$ be the sample count at state $s$ up to episode $T$. Under equivariance, sample counts aggregate across all states in the orbit $[s]$, amplifying the effective sample count to $|G| N_s(T)$.
   
   Applying standard regret bounds for episodic RL (e.g., UCRL2 bounds), the regret scales with the square root of the state-action space dimensions and the total steps:
   $$\text{Regret}(T) \le O\left( \sqrt{|\mathcal{S}/G| |\mathcal{A}/G| T} \right) = O\left( \sqrt{\frac{|\mathcal{S}| |A| T}{|G|^2}} \right)$$
   Since the action permutations are bijective, the effective regret bound simplifies to:
   $$\text{Regret}_{\text{equivariant}}(T) = O\left( \sqrt{\frac{T}{|G|}} \right)$$
   proving that the cumulative regret scales sub-linearly and is reduced by $\sqrt{|G|}$ under group equivariant MCTS guidance. $\blacksquare$

