"""Chromosome: hybrid real-valued multi-segment encoding.

genes.shape = (n_uavs, n_waypoints, 4)
    gene[...,0] = x         in [0, N-1]   waypoint x
    gene[...,1] = y         in [0, N-1]   waypoint y
    gene[...,2] = speed     in [v_min, v_max]  speed on the leg heading to this waypoint
    gene[...,3] = power     in [0, 1]     sensor power profile on that leg
Each UAV flies: base -> wp0 -> wp1 -> ... -> wpK-1 -> base.
(Waypoint ORDER is part of the genome => the GA also evolves the route.)
"""
import numpy as np


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None
        self.metrics = None

    @classmethod
    def random(cls, cfg, rng, sector_biased=False):
        U, K, N = cfg.n_uavs, cfg.n_waypoints, cfg.grid_size
        g = np.zeros((U, K, 4))
        if sector_biased:
            # each UAV gets an angular sector around the base -> good initial spread
            bx, by = cfg.base
            for u in range(U):
                lo, hi = (u / U) * np.pi / 2, ((u + 1) / U) * np.pi / 2
                th = rng.uniform(lo, hi, K)
                r = np.sort(rng.uniform(5, N * 1.1, K))
                g[u, :, 0] = bx + r * np.cos(th)
                g[u, :, 1] = by + r * np.sin(th)
        else:
            g[..., 0:2] = rng.uniform(0, N - 1, (U, K, 2))
        g[..., 2] = rng.uniform(cfg.v_min, cfg.v_max, (U, K))
        g[..., 3] = rng.uniform(0, 1, (U, K))
        ch = cls(g)
        ch.clip(cfg)
        return ch

    def copy(self):
        return Chromosome(self.genes.copy())

    def clip(self, cfg):
        g = self.genes
        g[..., 0:2] = np.clip(g[..., 0:2], 0, cfg.grid_size - 1)
        g[..., 2] = np.clip(g[..., 2], cfg.v_min, cfg.v_max)
        g[..., 3] = np.clip(g[..., 3], 0, 1)

    def route(self, u, base):
        """Full polyline of UAV u including base start/end."""
        wp = self.genes[u, :, :2]
        return np.vstack((base, wp, base))
