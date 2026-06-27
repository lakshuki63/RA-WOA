# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # No GUI — saves directly to file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

NUM_RUNS = 10
DIM      = 30
POP      = 30
ITERS    = 300

# ── Benchmarks ────────────────────────────────────────────────────────────────

def sphere(x):
    return np.sum(x**2)

def schwefel(x):
    """
    Schwefel function — the nastiest deceptive benchmark.
    Global min is at x=420.9687 in EVERY dimension (far from origin).
    WOA's spiral/encircle always drags agents toward current best,
    which is usually a deceptive local minimum near 0.
    RA-WOA's Levy arm can make large jumps toward the true basin.
    Global min = 0.
    """
    return 418.9829*DIM - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def shifted_rastrigin(x):
    SHIFT = np.full(DIM, 2.5)
    z = x - SHIFT
    return 10*DIM + np.sum(z**2 - 10*np.cos(2*np.pi*z))

BENCHMARKS = {
    "Sphere (Unimodal)":                 {"func": sphere,             "lb": -100,   "ub": 100},
    "Schwefel (Deceptive Multimodal)":   {"func": schwefel,           "lb": -500,   "ub": 500},
    "Shifted Rastrigin (Multimodal)":    {"func": shifted_rastrigin,  "lb": -5.12,  "ub": 5.12},
}

# ── Utilities ─────────────────────────────────────────────────────────────────
def levy(d, beta=1.5):
    num = gamma(1+beta)*np.sin(np.pi*beta/2)
    den = gamma((1+beta)/2)*beta*2**((beta-1)/2)
    su  = (num/den)**(1/beta)
    return np.random.normal(0,su,d) / (np.abs(np.random.normal(0,1,d))**(1/beta))

def diversity(pop):
    c = np.mean(pop, axis=0)
    return np.mean(np.linalg.norm(pop - c, axis=1))

# ── Standard WOA ─────────────────────────────────────────────────────────────
def woa(f, lb, ub):
    pop = np.random.uniform(lb, ub, (POP, DIM))
    fit = np.array([f(x) for x in pop])
    bi  = np.argmin(fit)
    bp, bf = pop[bi].copy(), fit[bi]
    curve = []
    for t in range(ITERS):
        a = 2 - 2*t/ITERS
        for i in range(POP):
            A = 2*a*np.random.random() - a
            C = 2*np.random.random()
            p = np.random.random()
            l = np.random.uniform(-1,1)
            if p < 0.5:
                if abs(A) < 1:
                    pop[i] = bp - A*np.abs(C*bp - pop[i])
                else:
                    r = np.random.randint(POP)
                    pop[i] = pop[r] - A*np.abs(C*pop[r] - pop[i])
            else:
                pop[i] = np.abs(bp-pop[i])*np.exp(l)*np.cos(2*np.pi*l)+bp
            pop[i] = np.clip(pop[i], lb, ub)
            fit[i] = f(pop[i])
            if fit[i] < bf:
                bp, bf = pop[i].copy(), fit[i]
        curve.append(bf)
    return bf, curve

# ── RA-WOA ───────────────────────────────────────────────────────────────────
def rawoa(f, lb, ub):
    pop = np.random.uniform(lb, ub, (POP, DIM))
    fit = np.array([f(x) for x in pop])
    bi  = np.argmin(fit)
    bp, bf = pop[bi].copy(), fit[bi]

    Q, N, Nt = np.zeros(3), np.zeros(3,int), 0
    D0 = diversity(pop)
    curve, arms = [], []

    for t in range(ITERS):
        Dt = diversity(pop)
        a  = np.clip(2*(1-t/ITERS)*(1+0.5*(D0-Dt)/(D0+1e-10)), 0, 3)
        ia = np.zeros(3, int)
        for i in range(POP):
            k = np.where(N==0)[0][0] if np.any(N==0) else np.argmax(Q+np.sqrt(2)*np.sqrt(np.log(Nt)/N))
            fo = fit[i]
            if k == 0:
                A = 2*a*np.random.random()-a
                if abs(A)<1:
                    pop[i] = bp - A*np.abs(2*np.random.random()*bp - pop[i])
                else:
                    r = np.random.randint(POP)
                    pop[i] = pop[r] - A*np.abs(2*np.random.random()*pop[r]-pop[i])
            elif k == 1:
                l = np.random.uniform(-1,1)
                pop[i] = np.abs(bp-pop[i])*np.exp(l)*np.cos(2*np.pi*l)+bp
            else:
                # Levy escape: jump to a random position scaled by Levy step
                # This is a REAL escape, not a tiny perturbation
                step = levy(DIM)
                # Scale step by the search space range so it's meaningful
                scale = (ub - lb) * 0.1
                pop[i] = pop[i] + scale * step / (np.abs(step).max() + 1e-10)
            pop[i] = np.clip(pop[i], lb, ub)
            fn = f(pop[i]); fit[i] = fn
            rw = max(0,(fo-fn)/(abs(fo)+1e-10))
            N[k]+=1; Nt+=1; Q[k]+=(rw-Q[k])/N[k]; ia[k]+=1
            if fn < bf: bp, bf = pop[i].copy(), fn
        curve.append(bf); arms.append(ia.copy())
    return bf, curve, np.array(arms)

# ── Run ───────────────────────────────────────────────────────────────────────
print("Running experiments...")
results = {}
for name, info in BENCHMARKS.items():
    f, lb, ub = info["func"], info["lb"], info["ub"]
    wc_all, rc_all, wf_all, rf_all, arm_all = [], [], [], [], []
    for run in range(NUM_RUNS):
        np.random.seed(run*17+3)
        wf, wc = woa(f, lb, ub)
        np.random.seed(run*17+3)
        rf, rc, ra = rawoa(f, lb, ub)
        wc_all.append(wc); rc_all.append(rc)
        wf_all.append(wf); rf_all.append(rf); arm_all.append(ra)
        tag = "[RA-WOA]" if rf < wf else "[WOA]   "
        print(f"  {name[:28]:28s} run{run+1:2d} WOA={wf:.2e} RA={rf:.2e} {tag}")
    results[name] = {
        "wc": np.array(wc_all), "rc": np.array(rc_all),
        "wf": np.array(wf_all), "rf": np.array(rf_all),
        "arms": np.array(arm_all), "lb": lb, "ub": ub
    }
    wm, rm = np.mean(wf_all), np.mean(rf_all)
    winner = "RA-WOA" if rm < wm else "WOA"
    pct = abs(wm-rm)/abs(wm+1e-30)*100
    print(f"  => WINNER: {winner}  ({pct:.1f}% gap)\n")

print("Generating plots...")

# ── Figure 1: Convergence + UCB1 arm budget ───────────────────────────────────
WOA_C, RA_C = '#e74c3c', '#2ecc71'
BG, AX = '#0d1117', '#161b22'
TXT, DIM_C = '#f0f6fc', '#8b949e'

names = list(results.keys())
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

for col, name in enumerate(names):
    res = results[name]
    x   = np.arange(ITERS)

    # Row 0: convergence
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(AX)
    wm_c = np.mean(res["wc"], axis=0)
    ws_c = np.std(res["wc"],  axis=0)
    rm_c = np.mean(res["rc"], axis=0)
    rs_c = np.std(res["rc"],  axis=0)
    ax.plot(x, wm_c, color=WOA_C, lw=2, label='WOA')
    ax.fill_between(x, np.maximum(wm_c-ws_c,1e-12), wm_c+ws_c, alpha=0.2, color=WOA_C)
    ax.plot(x, rm_c, color=RA_C,  lw=2, label='RA-WOA')
    ax.fill_between(x, np.maximum(rm_c-rs_c,1e-12), rm_c+rs_c, alpha=0.2, color=RA_C)
    ax.set_yscale('log')
    ax.set_title(name, color=TXT, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel("Iteration", color=DIM_C, fontsize=9)
    ax.set_ylabel("Best Fitness (log)", color=DIM_C, fontsize=9)
    ax.tick_params(colors=DIM_C, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.legend(fontsize=8, facecolor=AX, edgecolor='#30363d', labelcolor=TXT)
    ax.grid(True, alpha=0.12, color='white')

    wm, rm = np.mean(res["wf"]), np.mean(res["rf"])
    if rm < wm:
        label = f"RA-WOA WINS\n{(wm-rm)/abs(wm+1e-30)*100:.0f}% better"
        bc = '#1a3a2a'
    else:
        label = f"WOA WINS\n{(rm-wm)/abs(rm+1e-30)*100:.0f}% better"
        bc = '#3a1a1a'
    ax.text(0.97,0.97, label, transform=ax.transAxes, fontsize=8,
            va='top', ha='right', color=TXT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bc, edgecolor='#30363d', alpha=0.9))

    # Row 1: arm budget (stackplot)
    ax2 = fig.add_subplot(gs[1, col])
    ax2.set_facecolor(AX)
    am = np.mean(res["arms"], axis=0)            # (ITERS,3)
    tot = am.sum(axis=1, keepdims=True) + 1e-10
    prop = am / tot
    ax2.stackplot(x, prop.T,
                  labels=['Encircle (exploit)','Spiral (exploit)','Levy (explore)'],
                  colors=['#3498db','#e67e22','#9b59b6'], alpha=0.75)
    ax2.set_title(f"UCB1 Strategy Budget\n{name}", color=TXT, fontsize=9, fontweight='bold', pad=4)
    ax2.set_xlabel("Iteration", color=DIM_C, fontsize=9)
    ax2.set_ylabel("Fraction", color=DIM_C, fontsize=9)
    ax2.tick_params(colors=DIM_C, labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')
    ax2.legend(fontsize=7, facecolor=AX, edgecolor='#30363d', labelcolor=TXT, loc='upper right')
    ax2.grid(True, alpha=0.1, color='white')
    ax2.set_ylim(0,1)

import os
assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
os.makedirs(assets_dir, exist_ok=True)

fig.suptitle("RA-WOA vs WOA: Convergence and Strategy Adaptation",
             fontsize=15, fontweight='bold', color=TXT, y=0.99)
out1 = os.path.join(assets_dir, "rawoa_convergence.png")
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: Box plots ───────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor(BG)
for col, name in enumerate(names):
    res = results[name]
    ax  = axes2[col]
    ax.set_facecolor(AX)
    bp = ax.boxplot([res["wf"], res["rf"]], labels=["WOA","RA-WOA"],
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(color=DIM_C),
                    capprops=dict(color=DIM_C),
                    flierprops=dict(markerfacecolor=DIM_C, marker='o', markersize=4))
    bp['boxes'][0].set_facecolor(WOA_C); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(RA_C);  bp['boxes'][1].set_alpha(0.7)
    ax.set_title(name, color=TXT, fontsize=10, fontweight='bold')
    ax.set_ylabel("Final Fitness", color=DIM_C, fontsize=9)
    ax.tick_params(colors=DIM_C, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, alpha=0.15, color='white', axis='y')
    wm, rm = np.mean(res["wf"]), np.mean(res["rf"])
    w = "RA-WOA WINS" if rm < wm else "WOA WINS (expected)"
    ax.set_xlabel(f"=> {w}", color=DIM_C, fontsize=9)
fig2.suptitle(f"Final Fitness Distribution ({NUM_RUNS} runs) — Lower is Better",
              fontsize=13, fontweight='bold', color=TXT)
plt.tight_layout()
out2 = os.path.join(assets_dir, "rawoa_boxplot.png")
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {out2}")

# ── Print Table ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Benchmark':<34} {'Algo':<8} {'Mean':>12} {'Std':>12}  Winner")
print("="*70)
for name, res in results.items():
    wm,rm = np.mean(res["wf"]),np.mean(res["rf"])
    ws,rs = np.std(res["wf"]),np.std(res["rf"])
    winner = "RA-WOA" if rm < wm else "WOA"
    pct = abs(wm-rm)/abs(wm+1e-30)*100
    print(f"{name:<34} {'WOA':<8} {wm:>12.3e} {ws:>12.3e}")
    print(f"{'':<34} {'RA-WOA':<8} {rm:>12.3e} {rs:>12.3e}  => {winner} ({pct:.1f}% better)")
    print("-"*70)

print("\nDONE. Both PNGs saved to d:\\12345\\daa\\")
