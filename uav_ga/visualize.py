"""Plots: convergence, flight paths, coverage heat-maps, GA vs random."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def plot_convergence(history, path, random_curve=None):
    g = np.arange(len(history['best']))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(g, history['best'], lw=2, label="Best fitness")
    ax[0].plot(g, history['mean'], lw=2, label="Average fitness")
    if random_curve is not None:
        ax[0].plot(np.arange(len(random_curve)), random_curve, "k--", lw=1.5, label="Random search (best)")
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness")
    lo = min(min(history['mean'][3:] or [0]), 0) - 0.15
    ax[0].set_ylim(lo, max(history['best']) + 0.05)
    ax[0].set_title("Fitness convergence"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(g, history['diversity'], color="purple", label="Population diversity")
    ax2 = ax[1].twinx()
    ax2.plot(g, history['mut_rate'], color="gray", ls=":", label="Mutation rate")
    ax[1].set_xlabel("Generation"); ax[1].set_ylabel("Diversity"); ax2.set_ylabel("Mutation rate")
    ax[1].set_title("Diversity & adaptive mutation"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def draw_paths(ax, env, cfg, ch, title=""):
    n = cfg.grid_size
    ax.imshow(env.priority, origin="lower", cmap="YlOrRd", alpha=.55, vmin=1, vmax=6)
    ax.contourf(env.hazard, levels=[0.35, 0.6, 1.01], cmap="Purples", alpha=.45)
    nf = np.ma.masked_where(~env.no_fly, np.ones((n, n)))
    ax.imshow(nf, origin="lower", cmap="gray", alpha=.9)
    det = None
    for u in range(cfg.n_uavs):
        r = ch.route(u, np.array(cfg.base))
        ax.plot(r[:, 0], r[:, 1], "-o", color=COLORS[u % 5], lw=2, ms=4, label=f"UAV {u + 1}")
    ax.scatter(env.targets[:, 0], env.targets[:, 1], marker="x", c="k", s=18, label="Targets")
    ax.plot(*cfg.base, "k*", ms=16, label="Base")
    ax.set_xlim(-1, n); ax.set_ylim(-1, n); ax.set_title(title, fontsize=9)


def draw_heat(ax, ev, ch, title=""):
    cnt = ev.coverage_counts(ch)
    im = ax.imshow(cnt, origin="lower", cmap="inferno")
    ax.set_title(title, fontsize=9)
    return im


def plot_final(env, cfg, ev, ch, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    m = ch.metrics
    draw_paths(ax[0], env, cfg, ch, f"Evolved UAV paths | fitness {ch.fitness:.3f}")
    ax[0].legend(loc="lower right", fontsize=7)
    im = draw_heat(ax[1], ev, ch, f"Coverage heat-map | cov {m['coverage']:.2f}, det {m['detection']:.2f}")
    fig.colorbar(im, ax=ax[1], fraction=.046, label="# overlapping sensor footprints")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_generations(env, cfg, ev, snaps, path):
    gens = sorted(snaps)
    fig, ax = plt.subplots(2, len(gens), figsize=(4 * len(gens), 8))
    for j, g in enumerate(gens):
        ch = snaps[g]
        ev.evaluate(ch)
        draw_paths(ax[0, j], env, cfg, ch, f"Gen {g} | F={ch.fitness:.3f}")
        draw_heat(ax[1, j], ev, ch, f"Gen {g} | coverage={ch.metrics['coverage']:.2f}")
    fig.suptitle("Evolution of UAV paths (top) and coverage heat-maps (bottom)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_ga_vs_random(env, cfg, ev, ga_best, rs_best, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for r, (name, ch) in enumerate([("GA", ga_best), ("Random search", rs_best)]):
        draw_paths(ax[r, 0], env, cfg, ch, f"{name} | F={ch.fitness:.3f}")
        im = draw_heat(ax[r, 1], ev, ch, f"{name} coverage={ch.metrics['coverage']:.2f}, "
                                          f"det={ch.metrics['detection']:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
