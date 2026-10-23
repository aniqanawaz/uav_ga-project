"""Genetic operators: selection, crossover, mutation (spatially aware)."""
import numpy as np


# ---------------- Selection ----------------
def tournament_selection(pop, fits, k, rng):
    """Larger k  => higher selection pressure."""
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmax(fits[idx])]]


# ---------------- Crossover ----------------
def segment_swap(a, b, rng):
    """Swap whole UAV routes OR a waypoint segment inside each route."""
    c1, c2 = a.copy(), b.copy()
    U, K, _ = a.genes.shape
    if rng.random() < 0.5:
        m = rng.random(U) < 0.5
        if not m.any():
            m[rng.integers(U)] = True
        c1.genes[m], c2.genes[m] = b.genes[m].copy(), a.genes[m].copy()
    else:
        for u in range(U):
            i, j = sorted(rng.integers(0, K + 1, size=2))
            c1.genes[u, i:j], c2.genes[u, i:j] = b.genes[u, i:j].copy(), a.genes[u, i:j].copy()
    return c1, c2


def geometric_crossover(a, b, rng):
    """Arithmetic/geometric blend: child lies on the line between parent waypoints."""
    alpha = rng.random(a.genes.shape[:2] + (1,))
    c1, c2 = a.copy(), b.copy()
    c1.genes = alpha * a.genes + (1 - alpha) * b.genes
    c2.genes = (1 - alpha) * a.genes + alpha * b.genes
    return c1, c2


def crossover(a, b, rng, rate):
    if rng.random() > rate:
        return a.copy(), b.copy()
    return segment_swap(a, b, rng) if rng.random() < 0.5 else geometric_crossover(a, b, rng)


# ---------------- Mutation ----------------
def mutate(ch, rate, sigma, rng, cfg):
    g = ch.genes
    U, K, _ = g.shape
    # 1) gaussian shift of waypoint coordinates (annealed sigma)
    m = rng.random((U, K)) < rate
    g[..., 0:2] += m[..., None] * rng.normal(0, sigma, (U, K, 2))
    # 2) speed / sensor-power perturbation
    m2 = rng.random((U, K)) < rate
    g[..., 2] += m2 * rng.normal(0, 0.3, (U, K))
    g[..., 3] += m2 * rng.normal(0, 0.15, (U, K))
    # 3) order mutation: swap two waypoints in one route
    if rng.random() < rate:
        u = rng.integers(U)
        i, j = rng.choice(K, 2, replace=False)
        g[u, [i, j]] = g[u, [j, i]]
    # 4) random-reset waypoint (diversity injection / escape from local optimum)
    if rng.random() < rate * 0.5:
        u, k = rng.integers(U), rng.integers(K)
        g[u, k, 0:2] = rng.uniform(0, cfg.grid_size - 1, 2)
    return ch
