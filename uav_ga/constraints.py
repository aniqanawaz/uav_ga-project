"""Hard-constraint handling.

1. REPAIR  : waypoints inside no-fly cells are moved to the nearest open cell;
            routes longer than max_range are shrunk towards the base.
2. PENALTY : whatever repair cannot fix (e.g. a leg crossing a no-fly zone,
            time overrun) is punished in fitness.py with a dynamic penalty
            whose weight grows with the generation number.
"""
import numpy as np


def path_length(ch, u, base):
    r = ch.route(u, base)
    return float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())


def repair(ch, env, cfg, rng):
    base = np.array(cfg.base)
    ch.clip(cfg)

    # range repair (applied with probability -> keeps some infeasible diversity,
    # penalty function deals with the rest)
    if rng.random() < cfg.repair_range_prob:
        for u in range(cfg.n_uavs):
            for _ in range(15):
                if path_length(ch, u, base) <= cfg.max_range:
                    break
                ch.genes[u, :, :2] = base + (ch.genes[u, :, :2] - base) * 0.93

    # no-fly waypoint repair (always)
    n = cfg.grid_size
    for u in range(cfg.n_uavs):
        for k in range(cfg.n_waypoints):
            x, y = ch.genes[u, k, :2]
            ix, iy = int(round(min(max(x, 0), n - 1))), int(round(min(max(y, 0), n - 1)))
            if env.no_fly[iy, ix]:
                ch.genes[u, k, :2] = env.nearest_free(x, y)
    ch.clip(cfg)
    return ch
