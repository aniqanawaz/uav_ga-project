"""Genetic Algorithm engine (object oriented)."""
import numpy as np
from .chromosome import Chromosome
from .constraints import repair
from .operators import tournament_selection, crossover, mutate


class GeneticAlgorithm:
    def __init__(self, env, evaluator, cfg, snapshot_gens=None):
        self.env, self.ev, self.cfg = env, evaluator, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.snapshot_gens = set(snapshot_gens or [])
        self.history = dict(best=[], mean=[], worst=[], diversity=[], mut_rate=[])
        self.snapshots = {}
        self.best = None

    # ---- population initialisation (50% random, 50% sector-biased) ----
    def init_population(self):
        pop = []
        for i in range(self.cfg.pop_size):
            ch = Chromosome.random(self.cfg, self.rng, sector_biased=(i % 2 == 0))
            pop.append(repair(ch, self.env, self.cfg, self.rng))
        return pop

    def diversity(self, pop):
        G = np.stack([p.genes[..., :2].ravel() for p in pop])
        return float(G.std(axis=0).mean() / self.cfg.grid_size)

    def run(self, verbose=True):
        cfg = self.cfg
        pop = self.init_population()
        rate = cfg.mutation_rate
        stall, best_f = 0, -np.inf
        n_elite = max(1, int(cfg.elite_frac * cfg.pop_size))

        for gen in range(cfg.generations + 1):
            # dynamic penalty: gets stricter as evolution proceeds
            self.ev.penalty_scale = 1.0 + 1.0 * gen / cfg.generations
            for p in pop:
                self.ev.evaluate(p)          # re-evaluate (penalty scale changed)
            fits = np.array([p.fitness for p in pop])
            order = np.argsort(-fits)
            pop = [pop[i] for i in order]
            fits = fits[order]

            div = self.diversity(pop)
            self.history['best'].append(fits[0])
            self.history['mean'].append(fits.mean())
            self.history['worst'].append(fits[-1])
            self.history['diversity'].append(div)
            self.history['mut_rate'].append(rate)
            if gen in self.snapshot_gens:
                self.snapshots[gen] = pop[0].copy()
                self.snapshots[gen].fitness, self.snapshots[gen].metrics = pop[0].fitness, pop[0].metrics

            if fits[0] > best_f + 1e-5:
                best_f, stall = fits[0], 0
            else:
                stall += 1
            if self.best is None or fits[0] > self.best.fitness:
                self.best = pop[0].copy()
                self.best.fitness, self.best.metrics = pop[0].fitness, pop[0].metrics

            if verbose and gen % 10 == 0:
                print(f"gen {gen:3d} | best {fits[0]:.4f} | mean {fits.mean():.4f} "
                      f"| diversity {div:.3f} | mut {rate:.3f}")

            # ---- termination criteria ----
            if gen == cfg.generations or stall >= cfg.patience:
                if verbose and stall >= cfg.patience:
                    print(f"Early stop at gen {gen} (no improvement for {cfg.patience} gens)")
                break

            # adaptive mutation: raise when diversity collapses (premature convergence)
            rate = min(0.35, rate * 1.3) if div < cfg.diversity_threshold else \
                max(cfg.mutation_rate, rate * 0.9)
            sigma = cfg.sigma0 * (1 - 0.7 * gen / cfg.generations)   # annealed step

            # ---- next generation (elitism + selection + crossover + mutation + repair) ----
            nxt = [pop[i].copy() for i in range(n_elite)]
            while len(nxt) < cfg.pop_size:
                p1 = tournament_selection(pop, fits, cfg.tournament_size, self.rng)
                p2 = tournament_selection(pop, fits, cfg.tournament_size, self.rng)
                for c in crossover(p1, p2, self.rng, cfg.crossover_rate):
                    mutate(c, rate, sigma, self.rng, cfg)
                    repair(c, self.env, cfg, self.rng)
                    nxt.append(c)
            pop = nxt[:cfg.pop_size]

        self.ev.penalty_scale = 2.0
        self.ev.evaluate(self.best)
        return self.best, self.history


def random_search(env, evaluator, cfg, n_evals, seed=0):
    """Baseline: same evaluation budget, no evolution. Returns best-so-far curve per 'generation'."""
    rng = np.random.default_rng(seed)
    evaluator.penalty_scale = 2.0
    best, curve = None, []
    for i in range(n_evals):
        ch = repair(Chromosome.random(cfg, rng), env, cfg, rng)
        evaluator.evaluate(ch)
        if best is None or ch.fitness > best.fitness:
            best = ch
        if (i + 1) % cfg.pop_size == 0:
            curve.append(best.fitness)
    return best, curve
