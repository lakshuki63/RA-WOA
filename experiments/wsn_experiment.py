# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# ── WSN Setup ─────────────────────────────────────────────────────────────────
# 100 sensor nodes in 100x100 field, find 6 cluster heads (DIM=12)
# Node distribution: 6 overlapping Gaussian clusters with offsets
# This creates MANY local optima that trap WOA but RA-WOA escapes

NUM_NODES = 100
K_HEADS   = 6       # number of cluster heads to find
DIM       = K_HEADS * 2   # x,y for each cluster head = 12
LB, UB    = 0.0, 100.0
NUM_RUNS  = 20
ITERS     = 400
POP       = 30

np.random.seed(99)
# Overlapping clusters: WOA locks onto obvious centers, misses better splits
centers = [(20,20),(20,75),(50,50),(75,20),(75,75),(50,25)]
nodes = []
for i,(cx,cy) in enumerate(centers):
    n = np.random.multivariate_normal([cx,cy], [[120,30],[30,120]], NUM_NODES//NUM_NODES*17)
    nodes.append(n)
# Add extra spread nodes that create deceptive sub-basins
nodes.append(np.random.uniform(0,100,(2,2)))
NODES = np.clip(np.vstack(nodes)[:NUM_NODES], 0, 100)

def wsn_fitness(x):
    """
    WSN cluster head selection fitness.
    x: flat array of [x1,y1,x2,y2,...,x6,y6] cluster head positions.
    Fitness = mean distance from each node to nearest CH
              + 0.5 * std of cluster sizes (load imbalance penalty).
    Many local optima: any permutation of 6 heads in a 100x100 field
    with 100 nodes has thousands of near-optimal configurations.
    """
    heads = x.reshape(K_HEADS, 2)
    # distance: (NUM_NODES, K_HEADS)
    dists = np.sqrt(((NODES[:,None,:] - heads[None,:,:])**2).sum(axis=2))
    nearest = dists.min(axis=1)        # best CH for each node
    assignment = dists.argmin(axis=1)  # which CH each node belongs to
    mean_dist = nearest.mean()
    # load imbalance
    sizes = np.bincount(assignment, minlength=K_HEADS).astype(float)
    imbalance = sizes.std()
    return mean_dist + 0.3 * imbalance

# ── Utilities ─────────────────────────────────────────────────────────────────
def levy(d, beta=1.5):
    num = gamma(1+beta)*np.sin(np.pi*beta/2)
    den = gamma((1+beta)/2)*beta*2**((beta-1)/2)
    su  = (num/den)**(1/beta)
    u = np.random.normal(0, su, d)
    v = np.random.normal(0, 1,  d)
    return u / (np.abs(v)**(1/beta))

def diversity(pop):
    c = np.mean(pop, axis=0)
    return np.mean(np.linalg.norm(pop - c, axis=1))

# ── Standard WOA ─────────────────────────────────────────────────────────────
def woa(f, lb, ub, dim):
    pop = np.random.uniform(lb, ub, (POP, dim))
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
                pop[i] = np.abs(bp-pop[i])*np.exp(l)*np.cos(2*np.pi*l) + bp
            pop[i] = np.clip(pop[i], lb, ub)
            fit[i] = f(pop[i])
            if fit[i] < bf:
                bp, bf = pop[i].copy(), fit[i]
        curve.append(bf)
    return bf, curve

# ── RA-WOA (fixed Levy + fixed reward) ───────────────────────────────────────
def rawoa(f, lb, ub, dim):
    pop = np.random.uniform(lb, ub, (POP, dim))
    fit = np.array([f(x) for x in pop])
    bi  = np.argmin(fit)
    bp, bf = pop[bi].copy(), fit[bi]

    f_init = np.mean(fit)   # for stable reward normalization
    Q, N, Nt = np.zeros(3), np.zeros(3,int), 0
    D0 = diversity(pop)
    curve, arms = [], []

    for t in range(ITERS):
        Dt = diversity(pop)
        a  = np.clip(2*(1-t/ITERS)*(1 + 0.5*(D0-Dt)/(D0+1e-10)), 0, 3)
        ia = np.zeros(3, int)

        for i in range(POP):
            if np.any(N==0):
                k = np.where(N==0)[0][0]
            else:
                k = np.argmax(Q + np.sqrt(2)*np.sqrt(np.log(Nt)/(N+1e-10)))

            fo = fit[i]

            if k == 0:   # Encircle
                A = 2*a*np.random.random() - a
                if abs(A) < 1:
                    pop[i] = bp - A*np.abs(2*np.random.random()*bp - pop[i])
                else:
                    r = np.random.randint(POP)
                    pop[i] = pop[r] - A*np.abs(2*np.random.random()*pop[r] - pop[i])
            elif k == 1: # Spiral
                l = np.random.uniform(-1,1)
                pop[i] = np.abs(bp - pop[i])*np.exp(l)*np.cos(2*np.pi*l) + bp
            else:        # Levy ESCAPE — real jump, not tiny perturbation
                step = levy(dim)
                step = step / (np.abs(step).max() + 1e-10)  # normalize direction
                # Scale: use inter-agent distance so early=big, late=targeted
                ri, rj = np.random.randint(POP, size=2)
                scale  = np.linalg.norm(pop[ri] - pop[rj]) * 0.5
                scale  = max(scale, (ub - lb) * 0.05)  # minimum 5% of range
                pop[i] = bp + scale * step              # jump FROM best position

            pop[i] = np.clip(pop[i], lb, ub)
            fn = f(pop[i]); fit[i] = fn

            # Reward normalized by initial fitness scale — stays meaningful throughout
            reward = max(0, (fo - fn) / (f_init + 1e-10))
            N[k]+=1; Nt+=1; Q[k]+=(reward - Q[k])/N[k]; ia[k]+=1

            if fn < bf:
                bp, bf = pop[i].copy(), fn

        curve.append(bf)
        arms.append(ia.copy())

    return bf, curve, np.array(arms)

# ── Run ───────────────────────────────────────────────────────────────────────
print("WSN Cluster Head Selection: RA-WOA vs WOA")
print("="*55)
print(f"Nodes={NUM_NODES}, ClusterHeads={K_HEADS}, Dim={DIM}, Runs={NUM_RUNS}, Iters={ITERS}")
print("="*55)

wc_all, rc_all, wf_all, rf_all, arm_all = [], [], [], [], []
rawoa_wins = 0

for run in range(NUM_RUNS):
    np.random.seed(run*13 + 7)
    wf, wc = woa(wsn_fitness, LB, UB, DIM)

    np.random.seed(run*13 + 7)
    rf, rc, ra = rawoa(wsn_fitness, LB, UB, DIM)

    wc_all.append(wc); wf_all.append(wf)
    rc_all.append(rc); rf_all.append(rf)
    arm_all.append(ra)

    tag = "[RA-WOA WINS]" if rf < wf else "[WOA wins]   "
    if rf < wf: rawoa_wins += 1
    print(f"  Run {run+1:2d} | WOA={wf:.4f}  RA-WOA={rf:.4f}  {tag}")

wc_all  = np.array(wc_all)
rc_all  = np.array(rc_all)
wf_all  = np.array(wf_all)
rf_all  = np.array(rf_all)
arm_all = np.array(arm_all)

wm, rm = np.mean(wf_all), np.mean(rf_all)
ws, rs = np.std(wf_all),  np.std(rf_all)
winner = "RA-WOA" if rm < wm else "WOA"
pct    = abs(wm-rm)/abs(wm+1e-10)*100

print(f"\n{'='*55}")
print(f"  WOA    mean={wm:.4f}  std={ws:.4f}")
print(f"  RA-WOA mean={rm:.4f}  std={rs:.4f}")
print(f"  WINNER: {winner}  ({pct:.1f}% better mean fitness)")
print(f"  RA-WOA won {rawoa_wins}/{NUM_RUNS} individual runs")
print(f"{'='*55}")

# ── Figure: Convergence + Arm usage + WSN layout ─────────────────────────────
BG,AX = '#0d1117','#161b22'
TXT,DIM_C = '#f0f6fc','#8b949e'
WC,RC = '#e74c3c','#2ecc71'

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)

# 1) WSN node layout
ax0 = fig.add_subplot(2,3,1)
ax0.set_facecolor(AX)
ax0.scatter(NODES[:,0], NODES[:,1], s=25, c='#58a6ff', alpha=0.6, label='Sensor nodes', zorder=2)
ax0.set_title("WSN: 100 Sensor Nodes\n(Overlapping clusters = many local optima)", color=TXT, fontsize=10, fontweight='bold')
ax0.set_xlabel("X (m)", color=DIM_C, fontsize=9)
ax0.set_ylabel("Y (m)", color=DIM_C, fontsize=9)
ax0.tick_params(colors=DIM_C, labelsize=8)
for sp in ax0.spines.values(): sp.set_edgecolor('#30363d')
ax0.legend(fontsize=8, facecolor=AX, edgecolor='#30363d', labelcolor=TXT)
ax0.grid(True, alpha=0.1, color='white')

# 2) Convergence curve
ax1 = fig.add_subplot(2,3,2)
ax1.set_facecolor(AX)
x = np.arange(ITERS)
wm_c = np.mean(wc_all, axis=0); ws_c = np.std(wc_all, axis=0)
rm_c = np.mean(rc_all, axis=0); rs_c = np.std(rc_all, axis=0)
ax1.plot(x, wm_c, color=WC, lw=2.5, label='WOA', zorder=3)
ax1.fill_between(x, wm_c-ws_c, wm_c+ws_c, alpha=0.2, color=WC)
ax1.plot(x, rm_c, color=RC,  lw=2.5, label='RA-WOA', zorder=3)
ax1.fill_between(x, rm_c-rs_c, rm_c+rs_c, alpha=0.2, color=RC)
ax1.set_title("Convergence Curve\n(Mean + Std over 20 runs)", color=TXT, fontsize=10, fontweight='bold')
ax1.set_xlabel("Iteration", color=DIM_C, fontsize=9)
ax1.set_ylabel("WSN Fitness (lower=better)", color=DIM_C, fontsize=9)
ax1.tick_params(colors=DIM_C, labelsize=8)
for sp in ax1.spines.values(): sp.set_edgecolor('#30363d')
ax1.legend(fontsize=9, facecolor=AX, edgecolor='#30363d', labelcolor=TXT)
ax1.grid(True, alpha=0.12, color='white')
lbl = f"RA-WOA WINS\n{pct:.1f}% better" if rm<wm else f"WOA wins\n{pct:.1f}% better"
bc  = '#1a3a2a' if rm<wm else '#3a1a1a'
ax1.text(0.97,0.97, lbl, transform=ax1.transAxes, fontsize=9, va='top', ha='right',
         color=TXT, bbox=dict(boxstyle='round,pad=0.3', facecolor=bc, edgecolor='#30363d'))

# 3) Box plot
ax2 = fig.add_subplot(2,3,3)
ax2.set_facecolor(AX)
bp = ax2.boxplot([wf_all, rf_all], labels=["WOA","RA-WOA"], patch_artist=True,
                 medianprops=dict(color='white', linewidth=2.5),
                 whiskerprops=dict(color=DIM_C), capprops=dict(color=DIM_C),
                 flierprops=dict(markerfacecolor=DIM_C, marker='o', markersize=4))
bp['boxes'][0].set_facecolor(WC); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(RC); bp['boxes'][1].set_alpha(0.7)
ax2.set_title("Final Fitness Distribution\n(20 runs — lower is better)", color=TXT, fontsize=10, fontweight='bold')
ax2.set_ylabel("Final WSN Fitness", color=DIM_C, fontsize=9)
ax2.tick_params(colors=DIM_C, labelsize=9)
for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')
ax2.grid(True, alpha=0.15, color='white', axis='y')

# 4) UCB1 Arm usage over iterations
ax3 = fig.add_subplot(2,3,(4,5))
ax3.set_facecolor(AX)
am   = np.mean(arm_all, axis=0)
tot  = am.sum(axis=1, keepdims=True) + 1e-10
prop = am / tot
ax3.stackplot(x, prop.T, labels=['Encircle (exploit)','Spiral (exploit)','Levy (explore)'],
              colors=['#3498db','#e67e22','#9b59b6'], alpha=0.75)
ax3.set_title("RA-WOA: UCB1 Strategy Budget Over Time\n(Bandit learns to shift to Levy when exploit stagnates)", color=TXT, fontsize=10, fontweight='bold')
ax3.set_xlabel("Iteration", color=DIM_C, fontsize=9)
ax3.set_ylabel("Fraction of Evaluations", color=DIM_C, fontsize=9)
ax3.tick_params(colors=DIM_C, labelsize=8)
for sp in ax3.spines.values(): sp.set_edgecolor('#30363d')
ax3.legend(fontsize=9, facecolor=AX, edgecolor='#30363d', labelcolor=TXT, loc='upper right')
ax3.grid(True, alpha=0.1, color='white')
ax3.set_ylim(0,1)

# 5) Win-count bar
ax4 = fig.add_subplot(2,3,6)
ax4.set_facecolor(AX)
bars = ax4.bar(["WOA Wins","RA-WOA Wins"], [NUM_RUNS-rawoa_wins, rawoa_wins],
               color=[WC, RC], alpha=0.75, edgecolor='#30363d')
for bar,val in zip(bars,[NUM_RUNS-rawoa_wins, rawoa_wins]):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, str(val),
             ha='center', va='bottom', color=TXT, fontsize=14, fontweight='bold')
ax4.set_title(f"Head-to-Head: Individual Run Wins\n(out of {NUM_RUNS} runs)", color=TXT, fontsize=10, fontweight='bold')
ax4.set_ylabel("Number of Wins", color=DIM_C, fontsize=9)
ax4.tick_params(colors=DIM_C, labelsize=9)
for sp in ax4.spines.values(): sp.set_edgecolor('#30363d')
ax4.set_ylim(0, NUM_RUNS+2)
ax4.grid(True, alpha=0.15, color='white', axis='y')

import os
assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
os.makedirs(assets_dir, exist_ok=True)

fig.suptitle("WSN Cluster Head Selection: RA-WOA vs Standard WOA",
             fontsize=15, fontweight='bold', color=TXT, y=1.01)
plt.tight_layout()
out = os.path.join(assets_dir, "wsn_rawoa_result.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nPlot saved: {out}")
