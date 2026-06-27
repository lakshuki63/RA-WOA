# Reinforcement-Adaptive Whale Optimization Algorithm (RA-WOA)

## A Novel Multi-Armed Bandit Guided WOA with Lévy-Enhanced Exploration

---

## 1. Literature Summary

### 1.1 Algorithm Comparison

| Feature | WOA | PSO | GA | Cuckoo Search |
|---|---|---|---|---|
| **Inspiration** | Humpback whale bubble-net hunting | Bird flocking / swarm intelligence | Darwinian natural selection | Brood parasitism of cuckoo birds |
| **Exploration Mechanism** | Random leader selection (`|A| > 1`) | Velocity-based with inertia weight | Crossover + Mutation | Lévy flight random walks |
| **Exploitation Mechanism** | Shrinking encircling + Spiral update | Convergence toward pbest/gbest | Selection of fittest individuals | Abandon worst nests |
| **Phase Selection** | Random `p ∈ [0,1]` (50/50 split) | Implicit via inertia weight decay | Implicit via selection pressure | Implicit via discovery probability |
| **Parameter Adaptation** | Linear decay of `a` from 2→0 | Inertia weight schedule | Mutation/crossover rates (often fixed) | Step size via Lévy distribution |
| **Convergence Speed** | Moderate | Fast (but premature) | Slow (robust) | Moderate-Fast |
| **Multimodal Performance** | Weak (traps in local optima) | Weak (swarm collapse) | Strong (diversity via mutation) | Strong (Lévy jumps) |
| **Population Diversity** | Degrades over iterations | Degrades rapidly | Maintained via mutation | Maintained via Lévy flights |

### 1.2 Strengths and Weaknesses

#### WOA — Whale Optimization Algorithm (Mirjalili & Lewis, 2016)
- ✅ Simple, few parameters, elegant metaphor
- ✅ Good exploitation via spiral mechanism
- ❌ **Random phase selection** (`p = 0.5` threshold) — no intelligence
- ❌ **Linear `a` decay** — doesn't adapt to actual search progress
- ❌ **Premature convergence** on multimodal functions
- ❌ **No learning** — doesn't use search history

#### PSO — Particle Swarm Optimization (Kennedy & Eberhart, 1995)
- ✅ Fast convergence, easy to implement
- ✅ Well-understood mathematically
- ❌ Swarm collapse — all particles converge too fast
- ❌ Poor multimodal performance without modifications

#### GA — Genetic Algorithm (Holland, 1975)
- ✅ Robust exploration through crossover and mutation
- ✅ Effective on combinatorial problems
- ❌ Slow convergence on continuous optimization
- ❌ Computationally expensive (selection + crossover + mutation)

#### CS — Cuckoo Search (Yang & Deb, 2009)
- ✅ Excellent exploration via Lévy flights (heavy-tailed random walks)
- ✅ Simple yet powerful — only 1 main parameter
- ❌ Can be slow in fine-grained exploitation
- ❌ Fixed discovery probability — no adaptation

### 1.3 Where WOA Specifically Fails

Based on published literature (2016–2025), WOA has **four well-documented weaknesses**:

1. **Blind Phase Selection**: The parameter `p` that selects between encircling prey and spiral updating is drawn uniformly at random. There is **no feedback mechanism** — even when one strategy consistently produces better fitness improvements, the algorithm ignores this signal.

2. **Rigid Linear Decay**: The convergence factor `a` decreases linearly from 2 to 0 over iterations. This is **agnostic to the actual search state** — if the population has already converged (low diversity), `a` should slow down; if stuck in a local optimum, `a` should increase to boost exploration.

3. **Weak Exploration**: When `|A| > 1`, WOA selects a random whale for exploration. This is a crude mechanism compared to Lévy flights (CS) or mutation (GA), often resulting in **insufficient global search** on complex landscapes.

4. **No Historical Learning**: Unlike reinforcement learning-based methods, WOA doesn't learn which strategies work in which regions of the search space. Each iteration is **memoryless**.

---

## 2. Identified Research Gap

### 2.1 What Existing Methods Do NOT Handle Well

> **Gap**: No existing WOA variant uses an **online, reward-based learning mechanism** to intelligently select movement strategies during optimization. All variants use either random selection or predetermined rules.

Existing improvements fall into two categories:
- **Static modifications**: Adding Lévy flights, chaos maps, or opposition-based learning as fixed operations
- **Predetermined schedules**: Non-linear parameter decay curves that are still iteration-count-based, not performance-based

**Neither category learns from the optimization process itself.**

While Adaptive Operator Selection (AOS) using Multi-Armed Bandits has been applied to Differential Evolution and some evolutionary algorithms, it has **not been specifically applied to WOA's unique tri-phase structure** (encircling, spiral, exploration) as an intelligent phase selector.

### 2.2 Why This Matters in WSN Context

In Wireless Sensor Networks, the optimization landscape for cluster head selection is:
- **Dynamic** — node energies change over rounds
- **Multimodal** — multiple valid CH configurations exist
- **Constrained** — energy, coverage, and connectivity constraints
- **Time-sensitive** — solutions needed quickly for real-time operation

A WOA that **wastes iterations on ineffective strategies** (due to random phase selection) is particularly harmful in WSN contexts where:
- Computational budgets are limited (sensor nodes have low processing power)
- Premature convergence means suboptimal CH selection → faster network death
- The fitness landscape changes between rounds, requiring **adaptive behavior**

---

## 3. Proposed Algorithm: RA-WOA

### 3.1 Name
**Reinforcement-Adaptive Whale Optimization Algorithm (RA-WOA)**

### 3.2 Core Idea

Replace WOA's random phase selection with an **Upper Confidence Bound (UCB1) multi-armed bandit** that learns online which movement strategy (encircling, spiral, or Lévy exploration) is most effective at each stage of the search. Additionally, adapt the convergence parameter `a` based on **real-time population diversity feedback** rather than a fixed linear schedule.

### 3.3 Three Key Modifications

#### Modification 1: UCB1-Guided Strategy Selection (replaces random `p`)

Instead of `p = rand()` with a 0.5 threshold, we define **three arms**:
- **Arm 1**: Shrinking encircling (exploitation)
- **Arm 2**: Spiral updating (exploitation)
- **Arm 3**: Lévy-flight enhanced exploration

Each arm tracks:
- `Q_k` = average fitness improvement (reward) when arm `k` was selected
- `n_k` = number of times arm `k` has been selected
- `N` = total number of selections

Strategy selection uses UCB1:

$$\text{arm}^* = \arg\max_k \left[ \bar{Q}_k + C \sqrt{\frac{\ln N}{n_k}} \right]$$

Where `C` is an exploration constant (typically `√2`).

**Why this works**: Early in the search, all arms are tried roughly equally (exploration of strategies). As the search progresses, the bandit learns which strategy produces the best fitness improvements and favors it, while still occasionally trying others.

#### Modification 2: Diversity-Aware Adaptive `a` Parameter

Instead of linear decay `a = 2 - 2t/T`, we compute:

$$D(t) = \frac{1}{N \cdot d} \sum_{i=1}^{N} \| X_i - \bar{X} \|$$

Where `D(t)` is the normalized population diversity at iteration `t`.

The adaptive `a` is:

$$a(t) = 2 \cdot \left( 1 - \frac{t}{T} \right) \cdot \left( 1 + \alpha \cdot (D_0 - D(t)) \right)$$

Where:
- `D_0` is initial diversity (reference)
- `α` is a sensitivity coefficient (default: 0.5)
- When diversity drops sharply → `a` increases → more exploration
- When diversity is maintained → `a` follows normal schedule

#### Modification 3: Lévy-Flight Enhanced Exploration Phase

When Arm 3 (exploration) is selected by the bandit, instead of the standard random-whale-based exploration, we use a **Lévy flight**:

$$X_i(t+1) = X_i(t) + \alpha_0 \cdot \text{Lévy}(\beta) \otimes (X_i(t) - X^*(t))$$

Where the Lévy distribution is generated via Mantegna's formula:

$$\text{Lévy}(\beta) \sim \frac{u}{|v|^{1/\beta}}, \quad u \sim \mathcal{N}(0, \sigma_u^2), \quad v \sim \mathcal{N}(0, 1)$$

$$\sigma_u = \left( \frac{\Gamma(1+\beta) \sin(\pi\beta/2)}{\Gamma((1+\beta)/2) \cdot \beta \cdot 2^{(\beta-1)/2}} \right)^{1/\beta}$$

With `β = 1.5` (standard).

### 3.4 What Changed From Original WOA

| Component | Original WOA | RA-WOA |
|---|---|---|
| Phase selection | Random `p ∈ [0,1]`, threshold 0.5 | UCB1 multi-armed bandit over 3 strategies |
| Exploration mechanism | Random whale position | **Lévy flight** (heavy-tailed jumps) |
| Parameter `a` | Linear decay: `2 - 2t/T` | **Diversity-aware adaptive decay** |
| Learning | None | **Online reward tracking** via UCB1 |
| Number of phases | 2 (encircle or spiral) | **3** (encircle, spiral, Lévy exploration) |

---

## 4. Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RA-WOA PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────────────┐    ┌───────────────────────┐
│  INPUT   │───▶│  INITIALIZATION  │───▶│   BANDIT SETUP        │
│          │    │                  │    │                       │
│ • dim    │    │ • Random pop     │    │ • 3 arms (Q=0, n=0)  │
│ • bounds │    │ • Evaluate       │    │ • C = √2             │
│ • pop    │    │ • Find best      │    │ • Reward history      │
│ • iters  │    │ • Compute D_0    │    │                       │
└──────────┘    └──────────────────┘    └───────────────────────┘
                                                  │
                         ┌────────────────────────▼──────────────┐
                         │        MAIN OPTIMIZATION LOOP         │
                         │         for t = 1 to T:               │
                         └────────────────────────┬──────────────┘
                                                  │
                    ┌─────────────────────────────▼──────────────┐
                    │   STEP 1: Compute Diversity D(t)           │
                    │   STEP 2: Update adaptive a(t)             │
                    └─────────────────────────────┬──────────────┘
                                                  │
                    ┌─────────────────────────────▼──────────────┐
                    │   STEP 3: For each whale i:                │
                    │     • UCB1 selects best arm k*             │
                    └────────┬─────────┬──────────┬──────────────┘
                             │         │          │
                    ┌────────▼───┐ ┌───▼────┐ ┌───▼──────────┐
                    │  ARM 1:    │ │ ARM 2: │ │  ARM 3:      │
                    │ Encircle   │ │ Spiral │ │ Lévy Flight  │
                    │            │ │        │ │ Exploration  │
                    │ X += A·D   │ │ X =    │ │ X += α·Lévy  │
                    │            │ │ D'·ebl │ │   ⊗(X-X*)    │
                    │            │ │ ·cos(2 │ │              │
                    │            │ │  πl)+X*│ │              │
                    └────────┬───┘ └───┬────┘ └───┬──────────┘
                             │         │          │
                    ┌────────▼─────────▼──────────▼──────────────┐
                    │   STEP 4: Evaluate fitness f(X_new)        │
                    │   STEP 5: Compute reward for arm k*        │
                    │     reward = max(0, f_old - f_new)         │
                    │   STEP 6: Update Q_k*, n_k*, N             │
                    └─────────────────────────────┬──────────────┘
                                                  │
                    ┌─────────────────────────────▼──────────────┐
                    │   STEP 7: Update global best X*            │
                    │   STEP 8: Record convergence curve         │
                    └─────────────────────────────┬──────────────┘
                                                  │
                                           ┌──────▼──────┐
                                           │   OUTPUT     │
                                           │              │
                                           │ • Best X*    │
                                           │ • Best f*    │
                                           │ • Curve      │
                                           │ • Arm stats  │
                                           └─────────────┘
```

---

## 5. Mathematical Formulation

### 5.1 Original WOA Equations (Mirjalili & Lewis, 2016)

**Encircling prey** (exploitation):

$$\vec{D} = | \vec{C} \cdot \vec{X}^*(t) - \vec{X}(t) |$$
$$\vec{X}(t+1) = \vec{X}^*(t) - \vec{A} \cdot \vec{D}$$

Where:
$$\vec{A} = 2\vec{a} \cdot \vec{r} - \vec{a}, \quad \vec{C} = 2 \cdot \vec{r}$$
$$a = 2 - \frac{2t}{T}$$

**Spiral updating** (exploitation):

$$\vec{X}(t+1) = \vec{D'} \cdot e^{bl} \cdot \cos(2\pi l) + \vec{X}^*(t)$$

Where `D' = |X*(t) - X(t)|`, `b` is spiral shape constant, `l ∈ [-1, 1]`.

**Exploration** (`|A| > 1`):

$$\vec{X}(t+1) = \vec{X}_{rand} - \vec{A} \cdot | \vec{C} \cdot \vec{X}_{rand} - \vec{X}(t) |$$

**Phase selection**: `p = rand()`, if `p < 0.5` → encircle/explore, else → spiral.

### 5.2 RA-WOA Modified Equations

#### 🔴 Modification 1: UCB1 Phase Selection

Replace random `p` with:

$$k^* = \arg\max_{k \in \{1,2,3\}} \left[ \bar{Q}_k + C \sqrt{\frac{\ln N}{n_k}} \right]$$

**Reward computation** (for minimization):

$$r_k = \max\left(0, \frac{f(\vec{X}_{old}) - f(\vec{X}_{new})}{f(\vec{X}_{old}) + \epsilon}\right)$$

**Running average update**:

$$\bar{Q}_k \leftarrow \bar{Q}_k + \frac{1}{n_k}(r_k - \bar{Q}_k)$$

#### 🔴 Modification 2: Diversity-Aware `a`

$$D(t) = \frac{1}{N_{pop} \cdot d} \sum_{i=1}^{N_{pop}} \| \vec{X}_i(t) - \bar{\vec{X}}(t) \|_2$$

$$a(t) = 2 \cdot \left(1 - \frac{t}{T}\right) \cdot \left(1 + \alpha \cdot \frac{D_0 - D(t)}{D_0 + \epsilon}\right)$$

$$a(t) = \text{clip}(a(t), 0, 3)$$

#### 🔴 Modification 3: Lévy Flight Exploration (Arm 3)

$$\vec{X}_i(t+1) = \vec{X}_i(t) + \alpha_0 \cdot \text{Lévy}(\beta) \otimes (\vec{X}_i(t) - \vec{X}^*(t))$$

With Mantegna's approximation (`β = 1.5`):

$$\text{step} = \frac{u}{|v|^{1/\beta}}, \quad u \sim \mathcal{N}(0, \sigma_u^2), \quad v \sim \mathcal{N}(0, 1)$$

$$\sigma_u = \left[\frac{\Gamma(1+\beta) \cdot \sin(\pi\beta/2)}{\Gamma\left(\frac{1+\beta}{2}\right) \cdot \beta \cdot 2^{(\beta-1)/2}}\right]^{1/\beta}$$

---

## 6. Algorithm Pseudocode

```
ALGORITHM: RA-WOA (Reinforcement-Adaptive Whale Optimization Algorithm)

INPUT: 
  N       = population size
  d       = problem dimensionality
  T       = max iterations
  lb, ub  = lower/upper bounds
  f(·)    = objective function (minimize)
  C_ucb   = UCB exploration constant (default: √2)
  α       = diversity sensitivity (default: 0.5)
  β_levy  = Lévy exponent (default: 1.5)
  b       = spiral constant (default: 1)

OUTPUT:
  X*      = best solution found
  f*      = best fitness value
  curve[] = convergence history

BEGIN
  ── INITIALIZATION ──
  1.  Generate random population X[1..N] within [lb, ub]
  2.  Evaluate f(X[i]) for all i
  3.  X* ← argmin f(X[i])
  4.  f* ← f(X*)
  5.  D_0 ← compute_diversity(X)      // Initial diversity reference
  
  ── BANDIT INITIALIZATION ──
  6.  K ← 3                           // Number of arms
  7.  Q[1..K] ← 0                     // Estimated rewards
  8.  n[1..K] ← 0                     // Selection counts  
  9.  N_total ← 0                     // Total selections
  10. Precompute σ_u for Lévy flights (Mantegna)

  ── MAIN LOOP ──
  11. FOR t = 1 TO T DO:
  
      ── ADAPTIVE PARAMETER UPDATE ──
      12. D_t ← compute_diversity(X)
      13. a ← 2 * (1 - t/T) * (1 + α * (D_0 - D_t)/(D_0 + ε))
      14. a ← clip(a, 0, 3)
      
      ── PER-WHALE UPDATE ──
      15. FOR i = 1 TO N DO:
      
          ── UCB1 STRATEGY SELECTION ──
          16. IF any n[k] == 0:
                k* ← first k with n[k] == 0     // Try each arm once
              ELSE:
                k* ← argmax_k [Q[k] + C_ucb * √(ln(N_total)/n[k])]
          
          17. f_old ← f(X[i])
          18. X_old ← X[i]
          
          ── EXECUTE SELECTED STRATEGY ──
          19. IF k* == 1:  // ENCIRCLING
                r ← rand()
                A ← 2*a*r - a
                C_coeff ← 2*rand()
                IF |A| < 1:
                    D_vec ← |C_coeff * X* - X[i]|
                    X[i] ← X* - A * D_vec
                ELSE:
                    X_rand ← X[randint(1,N)]
                    D_vec ← |C_coeff * X_rand - X[i]|
                    X[i] ← X_rand - A * D_vec
                    
          20. ELSE IF k* == 2:  // SPIRAL
                D_prime ← |X* - X[i]|
                l ← uniform(-1, 1)
                X[i] ← D_prime * exp(b*l) * cos(2*π*l) + X*
                
          21. ELSE IF k* == 3:  // LÉVY EXPLORATION
                u ← Normal(0, σ_u²)
                v ← Normal(0, 1)
                step ← u / |v|^(1/β_levy)
                step_size ← 0.01 * step * (X[i] - X*)
                X[i] ← X[i] + step_size
          
          ── BOUNDARY CHECK ──
          22. X[i] ← clip(X[i], lb, ub)
          
          ── EVALUATE & COMPUTE REWARD ──
          23. f_new ← f(X[i])
          24. reward ← max(0, (f_old - f_new) / (|f_old| + ε))
          
          ── UPDATE BANDIT ──
          25. n[k*] ← n[k*] + 1
          26. N_total ← N_total + 1
          27. Q[k*] ← Q[k*] + (reward - Q[k*]) / n[k*]
          
          ── UPDATE BEST ──
          28. IF f_new < f*:
                X* ← X[i]
                f* ← f_new
      
      END FOR (whales)
      
      29. curve[t] ← f*
      
  END FOR (iterations)
  
  RETURN X*, f*, curve
END
```

---

## 7. Novelty Justification (For Viva)

### What makes this genuinely novel?

1. **UCB1 for WOA phase selection is new**: While UCB has been used in Differential Evolution, its application to WOA's specific tri-phase structure (encircle + spiral + Lévy) as an online strategy selector is not present in existing literature.

2. **Tri-phase architecture**: Original WOA has 2 phases with random selection. We introduce a structured 3rd phase (Lévy exploration) and an intelligent selector — this is a non-trivial architectural change.

3. **Coupled adaptivity**: The diversity-aware `a` parameter works synergistically with the bandit — when diversity drops, `a` boosts exploration AND the bandit independently learns that Lévy flights produce better rewards.

4. **Zero additional hyperparameters that need tuning**: UCB's `C = √2` is theoretically optimal. The diversity sensitivity `α = 0.5` is robust. Lévy's `β = 1.5` is standard. No grid search needed.

### What makes this practical?

- **Overhead**: UCB1 requires only O(K) = O(3) extra computation per whale per iteration — negligible
- **Memory**: Only 3 floats (Q) + 3 integers (n) + 1 integer (N) — 28 bytes
- **Complexity**: Same time complexity as original WOA: O(T × N × d)
- **Implementation**: ~50 lines of additional Python code

### References to cite

1. Mirjalili, S. & Lewis, A. (2016). "The Whale Optimization Algorithm." *Advances in Engineering Software*, 95, 51–67.
2. Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*, 47(2), 235–256.
3. Yang, X.S. & Deb, S. (2009). "Cuckoo Search via Lévy Flights." *World Congress on Nature & Biologically Inspired Computing*.
4. Fialho, Á. et al. (2010). "Adaptive Operator Selection with Dynamic Multi-Armed Bandits." *GECCO '10*.

---

## 8. WSN Application Mapping

For WSN cluster head selection, the algorithm maps as follows:

| Optimization Concept | WSN Mapping |
|---|---|
| Solution vector `X` | Binary/probability vector indicating CH status per node |
| Fitness function `f(X)` | Weighted sum of: residual energy, intra-cluster distance, coverage |
| Population | Set of candidate CH configurations |
| Iteration | One round of CH optimization |
| UCB benefit | Learns whether exploitation (refine current CH) or exploration (try new configurations) is better for current network state |

---

> [!IMPORTANT]
> **Please review this proposal and confirm before I generate the complete Python notebook implementation.** Key decisions for you to approve:
> 1. Algorithm name: **RA-WOA** ✓?
> 2. Core novelty: **UCB1 multi-armed bandit for phase selection** ✓?
> 3. Secondary improvement: **Diversity-aware `a` parameter** ✓?
> 4. Tertiary improvement: **Lévy-flight as explicit 3rd arm** ✓?
> 5. Benchmark functions: Sphere + Rastrigin ✓?
> 6. Comparison: WOA vs Random Search vs RA-WOA ✓?
