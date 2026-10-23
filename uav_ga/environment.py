"""Discretised hazard grid: no-fly zones, hazard pockets, priority zones, targets."""
import numpy as np


class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.grid_size
        rng = np.random.default_rng(cfg.env_seed)
        yy, xx = np.mgrid[0:n, 0:n]

        # --- No-fly zones (1 = blocked, 0 = open) ---
        self.no_fly = np.zeros((n, n), dtype=bool)
        self.no_fly[20:27, 22:32] = True      # restricted airspace
        self.no_fly[35:41, 8:14] = True       # tall buildings / airport

        # --- Hazard pockets (wind / radiation), values in [0,1] ---
        def gauss(cx, cy, s, a):
            return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))

        hz = gauss(30, 12, 5, 1.0) + gauss(12, 30, 4, 0.9) + gauss(38, 36, 5, 0.8)
        self.hazard = np.clip(hz, 0, 1)

        # --- Priority (disaster) zones: weight 1 (normal) ... ~5 (critical) ---
        pr = 1 + gauss(40, 42, 6, 4.0) + gauss(15, 20, 5, 3.0) + gauss(32, 6, 4, 2.0)
        self.priority = pr
        self.priority_total = pr.sum()

        # --- Hidden targets, more likely in high priority areas ---
        p = (pr ** 2).ravel()
        p /= p.sum()
        idx = rng.choice(n * n, size=cfg.n_targets, replace=False, p=p)
        self.targets = np.column_stack((idx % n, idx // n)).astype(int)   # (x, y)
        self.target_w = pr[self.targets[:, 1], self.targets[:, 0]]
        self.target_w_total = self.target_w.sum()

        free = np.argwhere(~self.no_fly)                 # (y, x)
        self._free_xy = free[:, ::-1].astype(float)      # (x, y)

    def nearest_free(self, x, y):
        """Closest open cell (used by the repair mechanism)."""
        d = (self._free_xy[:, 0] - x) ** 2 + (self._free_xy[:, 1] - y) ** 2
        return self._free_xy[np.argmin(d)]
