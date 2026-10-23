"""Uniqueness study: GA vs random search over several independent runs.
   python uniqueness.py   -> outputs/uniqueness.csv + outputs/fig6_uniqueness.png
"""
import itertools, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga.config import Config
from uav_ga.environment import Environment
from uav_ga.fitness import FitnessEvaluator
from uav_ga.ga import GeneticAlgorithm, random_search
from uav_ga import visualize as viz

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
N_RUNS = 5
cfg = Config(); env = Environment(cfg)

ga_sol, rs_sol = [], []
for s in range(1, N_RUNS + 1):
    c = Config(); c.seed = s
    ev = FitnessEvaluator(env, c)
    best, h = GeneticAlgorithm(env, ev, c).run(verbose=False)
    budget = c.pop_size * len(h['best'])
    ev2 = FitnessEvaluator(env, c)
    rs, _ = random_search(env, ev2, c, budget, seed=100 + s)
    ev.penalty_scale = ev2.penalty_scale = 2.0
    ev.evaluate(best); ev2.evaluate(rs)
    ga_sol.append((best, ev.coverage_counts(best) > 0))
    rs_sol.append((rs, ev2.coverage_counts(rs) > 0))
    print(f"run {s}: GA {best.fitness:.3f} | RS {rs.fitness:.3f}")

def jacc(a, b):
    return (a & b).sum() / (a | b).sum()

def pair_mean(A, B=None):
    if B is None:
        v = [jacc(a[1], b[1]) for a, b in itertools.combinations(A, 2)]
    else:
        v = [jacc(a[1], b[1]) for a in A for b in B]
    return float(np.mean(v))

rows = []
for name, sols in [("GA", ga_sol), ("Random search", rs_sol)]:
    f = np.array([s[0].fitness for s in sols])
    cov = np.array([s[0].metrics['coverage'] for s in sols])
    det = np.array([s[0].metrics['detection'] for s in sols])
    rows.append(dict(method=name, fitness_mean=f.mean(), fitness_std=f.std(),
                     fitness_best=f.max(), fitness_worst=f.min(),
                     coverage_mean=cov.mean(), detection_mean=det.mean(),
                     within_method_jaccard=pair_mean(sols)))
df = pd.DataFrame(rows)
df.loc[len(df)] = dict(method="GA vs RS (cross)", within_method_jaccard=pair_mean(ga_sol, rs_sol))
df.to_csv(f"{OUT}/uniqueness.csv", index=False)
print(df.to_string())

fig, ax = plt.subplots(2, N_RUNS, figsize=(3.2 * N_RUNS, 6.4))
for j in range(N_RUNS):
    for r, (nm, sols) in enumerate([("GA", ga_sol), ("RS", rs_sol)]):
        viz.draw_paths(ax[r, j], env, cfg, sols[j][0], f"{nm} run {j+1} | F={sols[j][0].fitness:.3f}")
fig.suptitle("Final solutions of 5 independent runs: GA (top) vs random search (bottom)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_uniqueness.png", dpi=120)
