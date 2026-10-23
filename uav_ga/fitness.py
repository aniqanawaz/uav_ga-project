"""Multi-objective fitness function with dynamic penalty for hard constraints.

F = w_cov*Coverage + w_det*Detection - w_en*Energy_n - w_t*Time_n - w_h*Hazard_n - Penalty

Coverage  : priority-weighted fraction of the grid inside sensor footprints
Detection : priority-weighted fraction of hidden targets that were sensed
Energy_n  : total energy / reference energy           (0..1)
Time_n    : mission time (slowest UAV) / max_time
Hazard_n  : hazard exposure along path / (U * max_range)
Penalty   : scale * ( range violation + time violation + no-fly crossings )
"""
import numpy as np


class FitnessEvaluator:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.penalty_scale = 1.0                 # increased by GA over generations
        self.n_evals = 0
        self.disks = {}
        for r in range(cfg.r_min, cfg.r_max + 1):
            d = np.array([(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                          if dx * dx + dy * dy <= r * r])
            self.disks[r] = d
        self.e_ref = (cfg.n_uavs * cfg.max_range * (cfg.e_base + cfg.e_speed * cfg.v_max ** 2)
                      + cfg.n_uavs * cfg.max_time * cfg.e_sensor)

    # ---- trace every leg of every UAV as sampled points ----
    def _legs(self, ch):
        cfg = self.cfg
        n = cfg.grid_size
        K = cfg.n_waypoints
        for u in range(cfg.n_uavs):
            route = ch.route(u, np.array(cfg.base))
            for i in range(K + 1):
                p0, p1 = route[i], route[i + 1]
                gi = min(i, K - 1)
                v, pw = ch.genes[u, gi, 2], ch.genes[u, gi, 3]
                d = float(np.linalg.norm(p1 - p0))
                k = max(2, int(np.ceil(d / cfg.step)) + 1)
                t = np.linspace(0, 1, k)[:, None]
                pts = p0 + (p1 - p0) * t
                ij = np.clip(np.rint(pts).astype(int), 0, n - 1)
                yield u, ij[:, 0], ij[:, 1], d, v, pw

    def evaluate(self, ch):
        cfg, env = self.cfg, self.env
        n = cfg.grid_size
        self.n_evals += 1
        mask = np.zeros(n * n, dtype=bool)
        energy = hazard = 0.0
        nofly_pts = 0
        L = np.zeros(cfg.n_uavs)
        T = np.zeros(cfg.n_uavs)

        for u, xs, ys, d, v, pw in self._legs(ch):
            t = d / v
            L[u] += d
            T[u] += t
            energy += d * (cfg.e_base + cfg.e_speed * v * v) + cfg.e_sensor * pw ** 2 * t
            hazard += env.hazard[ys, xs].mean() * d
            nofly_pts += int(env.no_fly[ys, xs].sum())
            r = int(round(cfg.r_min + pw * (cfg.r_max - cfg.r_min)))
            off = self.disks[r]
            cx = xs[:, None] + off[None, :, 0]
            cy = ys[:, None] + off[None, :, 1]
            ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
            mask[(cy * n + cx)[ok]] = True

        mask = mask.reshape(n, n)
        coverage = float((env.priority * mask).sum() / env.priority_total)
        det = float(env.target_w[mask[env.targets[:, 1], env.targets[:, 0]]].sum()
                    / env.target_w_total)
        energy_n = energy / self.e_ref
        time_n = float(T.max() / cfg.max_time)
        hazard_n = hazard / (cfg.n_uavs * cfg.max_range)

        range_viol = np.maximum(0, L - cfg.max_range).sum() / cfg.max_range
        time_viol = max(0.0, T.max() - cfg.max_time) / cfg.max_time
        penalty = self.penalty_scale * (cfg.pen_range * range_viol
                                        + cfg.pen_time * time_viol
                                        + cfg.pen_nofly * nofly_pts)

        fit = (cfg.w_cov * coverage + cfg.w_det * det - cfg.w_energy * energy_n
               - cfg.w_time * time_n - cfg.w_hazard * hazard_n - penalty)
        ch.fitness = float(fit)
        ch.metrics = dict(coverage=coverage, detection=det, energy=float(energy),
                          energy_n=float(energy_n), time=float(T.max()), hazard=float(hazard_n),
                          range_max=float(L.max()), nofly_pts=nofly_pts, penalty=float(penalty))
        return ch.fitness

    def coverage_counts(self, ch):
        """Heat-map: how many sensor footprints cover each cell (for visualisation)."""
        n = self.cfg.grid_size
        counts = np.zeros(n * n)
        for u, xs, ys, d, v, pw in self._legs(ch):
            r = int(round(self.cfg.r_min + pw * (self.cfg.r_max - self.cfg.r_min)))
            off = self.disks[r]
            cx = xs[:, None] + off[None, :, 0]
            cy = ys[:, None] + off[None, :, 1]
            ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
            np.add.at(counts, (cy * n + cx)[ok], 1)
        return counts.reshape(n, n)
