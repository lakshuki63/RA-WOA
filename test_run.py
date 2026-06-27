from ra_woa_core import *
import numpy as np

np.random.seed(42)
_, f1, _ = standard_woa(rastrigin, 30, -5.12, 5.12, 30, 500)
np.random.seed(42)
_, f2, _, s = ra_woa(rastrigin, 30, -5.12, 5.12, 30, 500)
print(f"Rastrigin (dim=30): WOA={f1:.4e}  RA-WOA={f2:.4e}")
print(f"Arm counts: {s['total_counts']}")
print(f"Q-values:   {s['Q_final']}")
