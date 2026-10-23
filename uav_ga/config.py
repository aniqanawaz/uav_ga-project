"""All tunable parameters live here (single source of truth)."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------- Environment ----------
    grid_size: int = 50              # 50 x 50 grid
    env_seed: int = 7                # fixes hazard/no-fly/priority layout
    n_targets: int = 30              # hidden targets to detect
    base: tuple = (2.0, 2.0)         # launch/landing station (x, y)

    # ---------- UAV fleet ----------
    n_uavs: int = 3
    n_waypoints: int = 6             # waypoints per UAV
    v_min: float = 1.0               # cells / time-unit
    v_max: float = 3.0
    r_min: int = 1                   # sensor radius (cells) at power = 0
    r_max: int = 4                   # sensor radius (cells) at power = 1
    step: float = 1.0                # path sampling step (cells)

    # ---------- Energy model ----------
    e_base: float = 1.0              # energy per cell flown (hover cost)
    e_speed: float = 0.15            # extra energy per cell ~ v^2 (drag)
    e_sensor: float = 0.6            # sensor energy per time-unit ~ power^2

    # ---------- Hard constraints ----------
    max_range: float = 110.0         # max path length per UAV (battery)
    max_time: float = 70.0           # max mission time per UAV

    # ---------- Fitness weights (maximise cov/det, minimise rest) ----------
    w_cov: float = 0.40
    w_det: float = 0.30
    w_energy: float = 0.10
    w_time: float = 0.05
    w_hazard: float = 0.15
    # penalty coefficients (hard constraints)
    pen_range: float = 2.0
    pen_time: float = 2.0
    pen_nofly: float = 0.10          # per sampled point inside no-fly zone
    repair_range_prob: float = 0.5   # chance to apply range-repair

    # ---------- GA hyper-parameters ----------
    pop_size: int = 60
    generations: int = 150
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    tournament_size: int = 3         # selection pressure
    elite_frac: float = 0.05
    sigma0: float = 4.0              # initial gaussian step for x,y
    patience: int = 60               # early stop if no improvement
    diversity_threshold: float = 0.06
    seed: int = 42
