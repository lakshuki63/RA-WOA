"""
RA-WOA: Reinforcement-Adaptive Whale Optimization Algorithm
Core implementations of Standard WOA, RA-WOA, and Random Search.
"""

import numpy as np
from scipy.special import gamma


# ============================================================
# BENCHMARK FUNCTIONS
# ============================================================

def sphere(x):
    """Sphere function - unimodal, simple bowl shape. Global min = 0 at origin."""
    return np.sum(x**2)

def rastrigin(x):
    """Rastrigin function - highly multimodal. Global min = 0 at origin."""
    d = len(x)
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    """Ackley function - multimodal with many local optima. Global min = 0."""
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20 + np.e

BENCHMARKS = {
    'Sphere':    {'func': sphere,    'lb': -100,  'ub': 100,  'opt': 0},
    'Rastrigin': {'func': rastrigin, 'lb': -5.12, 'ub': 5.12, 'opt': 0},
    'Ackley':    {'func': ackley,    'lb': -32,   'ub': 32,   'opt': 0},
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def levy_flight(d, beta=1.5):
    """Generate Levy flight step vector using Mantegna's algorithm."""
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, d)
    v = np.random.normal(0, 1, d)
    step = u / (np.abs(v) ** (1 / beta))
    return step

def compute_diversity(pop):
    """Population diversity: mean Euclidean distance from centroid."""
    centroid = np.mean(pop, axis=0)
    return np.mean(np.linalg.norm(pop - centroid, axis=1))


# ============================================================
# STANDARD WOA (Mirjalili & Lewis, 2016)
# ============================================================

def standard_woa(obj_func, dim, lb, ub, pop_size=30, max_iter=500):
    """Standard Whale Optimization Algorithm."""
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(x) for x in pop])
    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    curve = []
    b = 1  # spiral constant

    for t in range(max_iter):
        a = 2 - 2 * t / max_iter  # linear decay

        for i in range(pop_size):
            r1, r2 = np.random.random(), np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.random()
            l_val = np.random.uniform(-1, 1)

            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * best_pos - pop[i])
                    pop[i] = best_pos - A * D
                else:
                    ri = np.random.randint(pop_size)
                    D = np.abs(C * pop[ri] - pop[i])
                    pop[i] = pop[ri] - A * D
            else:
                D_prime = np.abs(best_pos - pop[i])
                pop[i] = D_prime * np.exp(b * l_val) * np.cos(2 * np.pi * l_val) + best_pos

            pop[i] = np.clip(pop[i], lb, ub)
            fitness[i] = obj_func(pop[i])

            if fitness[i] < best_fit:
                best_pos = pop[i].copy()
                best_fit = fitness[i]

        curve.append(best_fit)

    return best_pos, best_fit, curve


# ============================================================
# RA-WOA (Proposed: Reinforcement-Adaptive WOA)
# ============================================================

def ra_woa(obj_func, dim, lb, ub, pop_size=30, max_iter=500,
           C_ucb=None, alpha_div=0.5, beta_levy=1.5):
    """
    Reinforcement-Adaptive WOA with:
      1. UCB1 multi-armed bandit for strategy selection
      2. Diversity-aware adaptive 'a' parameter
      3. Levy-flight enhanced exploration arm
    """
    if C_ucb is None:
        C_ucb = np.sqrt(2)

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(x) for x in pop])
    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    # --- Bandit state ---
    K = 3  # arms: 0=encircle, 1=spiral, 2=levy
    Q = np.zeros(K)
    n_arms = np.zeros(K, dtype=int)
    N_total = 0

    D_0 = compute_diversity(pop)
    curve = []
    arm_counts_per_iter = []
    b = 1  # spiral constant

    for t in range(max_iter):
        # Adaptive a
        D_t = compute_diversity(pop)
        ratio = (D_0 - D_t) / (D_0 + 1e-10)
        a = 2 * (1 - t / max_iter) * (1 + alpha_div * ratio)
        a = np.clip(a, 0, 3)

        iter_arm_counts = np.zeros(K, dtype=int)

        for i in range(pop_size):
            # UCB1 selection
            if np.any(n_arms == 0):
                k = np.where(n_arms == 0)[0][0]
            else:
                ucb = Q + C_ucb * np.sqrt(np.log(N_total) / n_arms)
                k = np.argmax(ucb)

            f_old = fitness[i]

            if k == 0:  # ENCIRCLING
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * a * r1 - a
                C_coeff = 2 * r2
                if abs(A) < 1:
                    D_vec = np.abs(C_coeff * best_pos - pop[i])
                    pop[i] = best_pos - A * D_vec
                else:
                    ri = np.random.randint(pop_size)
                    D_vec = np.abs(C_coeff * pop[ri] - pop[i])
                    pop[i] = pop[ri] - A * D_vec

            elif k == 1:  # SPIRAL
                D_prime = np.abs(best_pos - pop[i])
                l_val = np.random.uniform(-1, 1)
                pop[i] = D_prime * np.exp(b * l_val) * np.cos(2 * np.pi * l_val) + best_pos

            elif k == 2:  # LEVY EXPLORATION
                step = levy_flight(dim, beta_levy)
                step_size = 0.01 * step * (pop[i] - best_pos)
                pop[i] = pop[i] + step_size

            pop[i] = np.clip(pop[i], lb, ub)
            f_new = obj_func(pop[i])
            fitness[i] = f_new

            # Reward & bandit update
            reward = max(0, (f_old - f_new) / (abs(f_old) + 1e-10))
            n_arms[k] += 1
            N_total += 1
            Q[k] += (reward - Q[k]) / n_arms[k]
            iter_arm_counts[k] += 1

            if f_new < best_fit:
                best_pos = pop[i].copy()
                best_fit = f_new

        curve.append(best_fit)
        arm_counts_per_iter.append(iter_arm_counts.copy())

    arm_stats = {
        'total_counts': n_arms.copy(),
        'Q_final': Q.copy(),
        'per_iter': np.array(arm_counts_per_iter),
    }
    return best_pos, best_fit, curve, arm_stats


# ============================================================
# RANDOM SEARCH (Baseline)
# ============================================================

def random_search(obj_func, dim, lb, ub, pop_size=30, max_iter=500):
    """Pure random search baseline."""
    best_pos = None
    best_fit = np.inf
    curve = []

    for _ in range(max_iter):
        batch = np.random.uniform(lb, ub, (pop_size, dim))
        for x in batch:
            f = obj_func(x)
            if f < best_fit:
                best_fit = f
                best_pos = x.copy()
        curve.append(best_fit)

    return best_pos, best_fit, curve
