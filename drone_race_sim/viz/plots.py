"""Static figures (matplotlib) for rollouts, controller comparisons and training curves."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

PHASE_COLORS = {"IDLE": "#9a9a9a", "AT_START": "#2f6db3", "OUTBOUND": "#2a9d4b", "TURNED": "#d1453b", "FINISHED": "#8e44ad"}
MODE_COLORS = {"backup": "#e67e22", "landing": "#c0392b", "hold": "#f1c40f"}
FOV_HALF = math.radians(55.4 / 2)


def draw_course(ax, geom, gate=True, fov_at=None):
    for (px, py), name in ((geom.pole_a_xy, "A"), (geom.pole_b_xy, "B")):
        if gate:
            ax.add_patch(Circle((px, py), geom.gate_radius_m, fill=False, ec="#b5b5b5", ls="--", lw=1))
        ax.add_patch(Circle((px, py), 0.25, fc="#222", ec="none", zorder=3))
        ax.annotate(f"Pole {name}", (px, py - geom.gate_radius_m - 0.22), ha="center", fontsize=8, color="#555")
    # tags face each other: little ticks on the side facing the other pole
    (ax_, ay_), (bx, by) = geom.pole_a_xy, geom.pole_b_xy
    ax.plot([ax_ + 0.25, ax_ + 0.45], [ay_, ay_], color="#e8a317", lw=3, zorder=4)
    ax.plot([bx - 0.25, bx - 0.45], [by, by], color="#e8a317", lw=3, zorder=4)
    if fov_at is not None:
        x, y, yaw, rng = fov_at
        ax.add_patch(Wedge((x, y), rng, math.degrees(yaw - FOV_HALF), math.degrees(yaw + FOV_HALF), fc="#2f6db3", alpha=0.08, ec="none"))
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, geom.distance + 1.3)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("X: Pole A to Pole B (m)")
    ax.set_ylabel("Y left (m)")
    ax.grid(alpha=0.25)


def _segments(ax, x, y, colors, lw=1.6, alpha=0.9):
    for k in range(len(x) - 1):
        ax.plot(x[k:k + 2], y[k:k + 2], color=colors[k], lw=lw, alpha=alpha, solid_capstyle="round")


def plot_rollout(ro, i: int, geom, path: Path, title: str | None = None) -> None:
    """One run: top-down path (phase colours, estimate dashed), height, sticks, tag visibility."""
    tr = ro.tracks[i]
    t, x, y, z = tr["t"], tr["x"], tr["y"], tr["z"]
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    k0 = len(x) // 3
    draw_course(ax, geom, fov_at=(x[k0], y[k0], tr["yaw"][k0], 3.0))
    cols = [MODE_COLORS.get(m.split(":")[0]) or PHASE_COLORS[p] for p, m in zip(tr["phase"], tr["mode"])]
    _segments(ax, x, y, cols)
    if ro.meta.get("sensing") == "realistic":
        ax.plot(tr["est_x"], tr["est_y"], color="#333", lw=0.8, ls=":")
        lost = tr["lost"]
        ax.scatter(x[lost][::6], y[lost][::6], s=9, c="#c0392b", zorder=5)
    for k in range(0, len(x), 40):   # heading arrows
        ax.arrow(x[k], y[k], 0.16 * math.cos(tr["yaw"][k]), 0.16 * math.sin(tr["yaw"][k]), head_width=0.06, color="#111", zorder=6)
    ax.plot(x[0], y[0], "o", color="white", mec="k", zorder=7)
    outcome = ro.outcome[i] + (f" ({ro.land_reason[i]})" if ro.land_reason[i] else "")
    lap = ro.lap_time_s[i]
    ax.set_title(title or f"{ro.controller} | {ro.scenario} | seed {ro.seeds[i]} | {outcome}" + (f" | lap {lap:.1f} s" if lap else ""),
                 fontsize=11)
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=k.lower().replace("_", " ")) for k, c in PHASE_COLORS.items() if k != "IDLE"]
    handles += [plt.Line2D([0], [0], color=c, lw=3, label=k) for k, c in MODE_COLORS.items() if any(m.startswith(k) for m in tr["mode"])]
    if ro.meta.get("sensing") == "realistic":
        handles += [plt.Line2D([0], [0], color="#333", lw=1, ls=":", label="estimate"),
                    plt.Line2D([0], [0], marker="o", color="w", mfc="#c0392b", ms=5, label="no tag fix (dead reckoning)"),
                    matplotlib.patches.Patch(fc="#2f6db3", alpha=0.15, label="camera FOV (55 deg)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    a2 = fig.add_subplot(gs[1, 0])
    a2.plot(t, z, color="#2f6db3")
    a2.axhspan(0.4, 2.5, color="#2a9d4b", alpha=0.07)
    a2.set_ylabel("height (m)"); a2.set_ylim(0, 3); a2.grid(alpha=0.25); a2.set_title("altitude (green = allowed band)", fontsize=9)
    a3 = fig.add_subplot(gs[1, 1], sharex=a2)
    for k, (name, c) in enumerate((("right", "#d98200"), ("forward", "#2a9d4b"), ("up", "#888"), ("yaw", "#d1453b"))):
        a3.plot(t, tr["rc"][:, k], color=c, lw=0.9, label=name)
    a3.set_ylabel("rc stick"); a3.set_ylim(-105, 105); a3.grid(alpha=0.25); a3.legend(ncol=4, fontsize=7, loc="upper right")
    a3.set_title("commands sent", fontsize=9)
    a4 = fig.add_subplot(gs[1, 2], sharex=a2)
    if ro.meta.get("sensing") == "realistic":
        a4.fill_between(t, 0, tr["seen"][:, 0], step="mid", color="#e8a317", alpha=0.85, label="tag A")
        a4.fill_between(t, 1.2, 1.2 + tr["seen"][:, 1], step="mid", color="#2f6db3", alpha=0.85, label="tag B")
        a4.set_yticks([]); a4.legend(fontsize=7, loc="upper right", ncol=2); a4.set_title("tag detections", fontsize=9)
    else:
        speed = np.hypot(np.gradient(x, t), np.gradient(y, t))
        a4.plot(t, speed, color="#333"); a4.set_ylabel("speed (m/s)"); a4.grid(alpha=0.25); a4.set_title("ground speed", fontsize=9)
    for a in (a2, a3, a4):
        a.set_xlabel("time (s)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(rollouts: list, geom, path: Path, title: str, max_paths: int = 25) -> None:
    """Rows = controllers: many paths overlaid (phase colours) + outcome summary in the title."""
    n = len(rollouts)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n), squeeze=False)
    for ax, ro in zip(axes[:, 0], rollouts):
        draw_course(ax, geom)
        for tr in ro.tracks[:max_paths]:
            cols = [MODE_COLORS.get(m.split(":")[0]) or PHASE_COLORS[p] for p, m in zip(tr["phase"], tr["mode"])]
            _segments(ax, tr["x"], tr["y"], cols, lw=1.0, alpha=0.5)
        s = ro.summary()
        lap = f", median lap {s['median_lap_s']:.1f} s" if s["median_lap_s"] else ""
        ax.set_title(f"{ro.controller}: {s['outcomes']}{lap}" + (f", {s['takeovers']} backup takeovers" if s["takeovers"] else ""),
                     fontsize=9)
        ax.set_xlabel("")
    axes[-1, 0].set_xlabel("X: Pole A to Pole B (m)    green = outbound, red = return, orange = Maude backup, dark red = landing")
    fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(rows: dict[str, dict[str, dict]], path: Path, title: str) -> None:
    """Stacked bars per scenario and controller: share of lap / landed / crash / timeout."""
    scen = list(next(iter(rows.values())).keys())
    ctrls = list(rows.keys())
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(scen) * len(ctrls) / 2), 4.2))
    w = 0.8 / len(ctrls)
    colors = {"lap": "#2a9d4b", "lap_est_only": "#9ad6a8", "landed": "#e8a317", "timeout": "#9a9a9a", "crash": "#c0392b"}
    for ci, c in enumerate(ctrls):
        for si, sc in enumerate(scen):
            oc = rows[c][sc]["outcomes"]
            n = sum(oc.values())
            bottom = 0.0
            for key in ("lap", "lap_est_only", "landed", "timeout", "crash"):
                v = sum(cnt for o, cnt in oc.items() if o == key or (key == "crash" and o.startswith("crash"))) / n
                ax.bar(si + ci * w - 0.4 + w / 2, v, w * 0.92, bottom=bottom, color=colors[key], label=key if (si, ci) == (0, 0) else None)
                bottom += v
            ax.text(si + ci * w - 0.4 + w / 2, 1.02, c[:12], rotation=90, ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(scen)), scen)
    ax.set_ylim(0, 1.35); ax.set_yticks([0, .5, 1]); ax.set_ylabel("share of runs")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_training(log_csv: Path, path: Path, title: str = "PPO training") -> None:
    rows = list(csv.DictReader(open(log_csv)))
    f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])
    it = f("iter")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    axes[0, 0].plot(it, f("success"), color="#2a9d4b"); axes[0, 0].set_title("lap success rate")
    axes[0, 1].plot(it, f("lap_time_s"), color="#2f6db3"); axes[0, 1].set_title("median lap time (s)")
    axes[0, 2].plot(it, f("ep_return"), color="#333"); axes[0, 2].set_title("episode return")
    axes[1, 0].plot(it, f("dr_level"), color="#d98200"); axes[1, 0].set_title("domain-randomization level")
    for k, c in (("cause_pole", "#c0392b"), ("cause_ground", "#8e44ad"), ("cause_oob", "#2f6db3"), ("cause_timeout", "#9a9a9a")):
        if k in rows[0]:
            axes[1, 1].plot(it, f(k), label=k.replace("cause_", ""), color=c)
    axes[1, 1].set_title("terminations (share of episodes)"); axes[1, 1].legend(fontsize=7)
    axes[1, 2].plot(it, f("kl"), color="#555"); axes[1, 2].set_title("KL (policy step size)")
    for ax in axes.ravel():
        ax.grid(alpha=0.25); ax.set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
