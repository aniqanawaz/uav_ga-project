"""Hyper-parameter sensitivity analysis:  python experiments.py"""
import copy, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga.config import Config
from uav_ga.environment import Environment
from uav_ga.fitness import FitnessEvaluator
from uav_ga.ga import GeneticAlgorithm

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
SEEDS = [1, 2, 3]
GENS = 50
GRID = {
    "pop_size": [20, 60, 120],
    "mutation_rate": [0.02, 0.15, 0.40],
    "tournament_size": [2, 5, 12],
}

rows, curves = [], {}
for param, values in GRID.items():
    for v in values:
        finals, cs = [], []
        for s in SEEDS:
            cfg = Config(); setattr(cfg, param, v)
            cfg.generations, cfg.seed, cfg.patience = GENS, s, 999
            env = Environment(cfg); ev = FitnessEvaluator(env, cfg)
            best, h = GeneticAlgorithm(env, ev, cfg).run(verbose=False)
            finals.append(best.fitness); cs.append(h['best'])
        rows.append(dict(param=param, value=v, mean_fitness=np.mean(finals), std=np.std(finals)))
        curves[(param, v)] = np.mean(cs, axis=0)
        print(param, v, f"{np.mean(finals):.4f} +- {np.std(finals):.4f}")

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/sensitivity.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, p in zip(ax, GRID):
    for v in GRID[p]:
        a.plot(curves[(p, v)], label=f"{p}={v}")
    a.set_title(p); a.set_xlabel("Generation"); a.set_ylabel("Best fitness (mean of 3 seeds)")
    a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_sensitivity.png", dpi=150)
print(df)
