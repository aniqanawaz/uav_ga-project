"""Run the full UAV GA simulation:  python main.py"""
import json, os, time
from uav_ga.config import Config
from uav_ga.environment import Environment
from uav_ga.fitness import FitnessEvaluator
from uav_ga.ga import GeneticAlgorithm, random_search
from uav_ga import visualize as viz

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

cfg = Config()
env = Environment(cfg)
ev = FitnessEvaluator(env, cfg)

t0 = time.time()
snap_gens = [0, cfg.generations // 4, cfg.generations // 2, cfg.generations]
ga = GeneticAlgorithm(env, ev, cfg, snapshot_gens=snap_gens)
best, hist = ga.run()
print(f"GA finished in {time.time() - t0:.1f}s | evaluations: {ev.n_evals}")

# ---- baseline random search with the SAME evaluation budget ----
budget = cfg.pop_size * (len(hist['best']))
ev_rs = FitnessEvaluator(env, cfg)
rs_best, rs_curve = random_search(env, ev_rs, cfg, budget, seed=1)

# ---- plots ----
viz.plot_convergence(hist, f"{OUT}/fig1_convergence.png", rs_curve)
viz.plot_final(env, cfg, ev, best, f"{OUT}/fig2_final_paths_heatmap.png")
snaps = ga.snapshots
if cfg.generations not in snaps or len(hist['best']) - 1 not in snaps:
    snaps[len(hist['best']) - 1] = best      # early-stop case
viz.plot_generations(env, cfg, ev, snaps, f"{OUT}/fig3_generations.png")
viz.plot_ga_vs_random(env, cfg, ev, best, rs_best, f"{OUT}/fig4_ga_vs_random.png")

res = dict(ga=dict(fitness=best.fitness, **best.metrics),
           random_search=dict(fitness=rs_best.fitness, **rs_best.metrics))
json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
