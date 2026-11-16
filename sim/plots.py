"""Figures for one results folder.

    python -m sim.plots results/<date_time>

Writes <folder>/figures/: regret.png, value.png, approaches.png, regret_by_landscape.png, positions.png,
and distributions/<adviser>_L<landscape>.png (where each group works, at several periods). No captions on the figures:
  regret of AI users      cost of their move minus the cost of their best option (0 = best possible move)
  value per researcher    significance minus crowding, kappa_t(x) - c m_t(x), where the researcher works
  approaches used         approaches holding at least 1% of the group

Per period t and adviser, with the crowd that actually followed (m_{t+1}), the cost of a move is the spec's
C_t(x, y) = gamma d(x, y) - kappa_{t+1}(y) + c m_{t+1}(y) (section 11; lower is better).
  AI cost      the cost of the moves the AI users made, averaged over AI users
  AI regret    AI cost minus the cost of the best of their 12 options, averaged over AI users
  non-AI cost  the cost of the best option, averaged over non-AI users (they are at the period equilibrium)
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .parameters import C, GAMMA
from .world import DISTANCE, N, OPTIONS, QUESTION_TYPES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ADVISERS = ["naive", "haiku_thinking", "haiku", "stay", "random"]
LABELS = {"naive": "Naive", "haiku_thinking": "Haiku, thinking on", "haiku": "Haiku, thinking off",
          "stay": "Stay", "random": "Random"}
COLORS = {"naive": "#2a78d6", "haiku_thinking": "#eb6834", "haiku": "#1baf7a", "stay": "#eda100", "random": "#e87ba4"}
INK, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "font.size": 9, "legend.frameon": False, "figure.dpi": 150})


def present(df):
    """The advisers that are in this results folder, in the fixed order."""
    return [a for a in ADVISERS if a in set(df.adviser)]


def measures(folder):
    """One row per adviser, landscape and period t."""
    rows = []
    for run in sorted(Path(folder).glob("*_theta*")):
        adviser = run.name.split("_theta")[0]
        landscape = int(run.name.split("_L")[1][:2])
        d = pd.read_csv(run / "distributions.csv").sort_values(["t", "x"])
        at = {t: g for t, g in d.groupby("t")}
        has_calls = (run / "calls.jsonl").stat().st_size > 0  # no calls when theta = 0
        calls = pd.read_json(run / "calls.jsonl", lines=True) if has_calls else pd.DataFrame(columns=["t", "x", "y", "mass"])
        for t in range(d.t.max()):
            nxt = at[t + 1]
            cost = GAMMA * DISTANCE - nxt.kappa.to_numpy()[None] + C * nxt.m.to_numpy()[None]
            best = np.array([cost[x, OPTIONS[x]].min() for x in range(N)])
            c = calls[calls.t == t]
            share = (c.mass / c.mass.sum()).to_numpy(float)
            made = cost[c.x.astype(int), c.y.astype(int)]
            ai_cost = (share * made).sum() if len(c) else np.nan
            ai_regret = (share * (made - best[c.x.astype(int)])).sum() if len(c) else np.nan
            rows.append({"adviser": adviser, "landscape": landscape, "t": t,
                         "ai_cost": ai_cost, "ai_regret": ai_regret,
                         "non_ai_cost": (at[t].mN.to_numpy() * best).sum(),
                         "ai_approaches": int((nxt.mA > 0).sum()), "H_ai": (nxt.mA ** 2).sum(),
                         "H_non_ai": (nxt.mN ** 2).sum(),
                         "value_ai": (nxt.mA * (nxt.kappa - C * nxt.m)).sum(),
                         "value_non_ai": (nxt.mN * (nxt.kappa - C * nxt.m)).sum(),
                         "used_ai": int((nxt.mA >= 0.01).sum()), "used_non_ai": int((nxt.mN >= 0.01).sum())})
    return pd.DataFrame(rows)


def over_time(ax, df, column):
    """Mean over landscapes with a band of +- 1 SD; whole periods."""
    for adviser in present(df):
        g = df[df.adviser == adviser].groupby("t")[column]
        mean, sd = g.mean(), g.std()
        ax.plot(mean.index + 1, mean, color=COLORS[adviser], lw=2, label=LABELS[adviser])
        ax.fill_between(mean.index + 1, mean - sd, mean + sd, color=COLORS[adviser], alpha=0.12, lw=0)
    t_max = df.t.max() + 1
    ax.set_xticks(range(1, t_max + 1))
    ax.set_xlim(0.7, t_max + 0.3)
    ax.set_xlabel("t (period)")


def regret_figure(df, out):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    over_time(ax, df, "ai_regret")
    ax.set_ylabel("regret (0 = best possible move)")
    ax.set_title("Regret of AI users", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "regret.png")


def value_figure(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    over_time(axes[0], df, "value_ai")
    over_time(axes[1], df, "value_non_ai")
    axes[0].set_title("AI users", loc="left")
    axes[1].set_title("Non-AI users", loc="left")
    axes[0].set_ylabel("value per researcher (higher is better)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "value.png")


def approaches_figure(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    over_time(axes[0], df, "used_ai")
    over_time(axes[1], df, "used_non_ai")
    axes[0].set_title("Approaches used by AI users", loc="left")
    axes[1].set_title("Approaches used by non-AI users", loc="left")
    axes[0].set_ylabel("approaches with at least 1% of the group")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "approaches.png")


def landscape_figure(df, out):
    advisers = present(df)
    per = df.groupby(["adviser", "landscape"]).ai_regret.mean().unstack(0)[advisers]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = np.arange(len(advisers))
    for landscape, row in per.iterrows():
        ax.plot(xs, row.to_numpy(), color=GRID, lw=1.2, zorder=1)
        ax.annotate(f"landscape {landscape}", (xs[-1], row.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=INK, va="center")
    for i, adviser in enumerate(advisers):
        ax.scatter(np.full(len(per), i), per[adviser], s=40, color=COLORS[adviser], zorder=2,
                   edgecolor="white", linewidth=1.5)
    ax.set_xticks(xs, [LABELS[a].replace(", ", ",\n") for a in advisers])
    ax.set_xlim(-0.4, len(advisers) - 0.2)
    ax.set_ylabel("regret (0 = best possible move)")
    ax.set_title("Regret of AI users on each landscape", loc="left")
    fig.tight_layout()
    fig.savefig(out / "regret_by_landscape.png")


def position_figure(folder, out):
    models = [a for a in ("haiku_thinking", "haiku") if any(Path(folder).glob(f"{a}_theta*"))]
    if not models:  # no Haiku runs in this folder
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / len(models)
    for k, adviser in enumerate(models):
        positions = []
        for run in Path(folder).glob(f"{adviser}_theta*"):
            positions += [c["position"] for c in map(json.loads, open(run / "calls.jsonl")) if c["position"]]
        share = np.bincount(positions, minlength=13)[1:] / len(positions)
        ax.bar(np.arange(1, 13) + (k - (len(models) - 1) / 2) * width, share, width=width - 0.02, color=COLORS[adviser],
               label=LABELS[adviser])
    ax.axhline(1 / 12, color=INK, lw=1, ls="--", label="no bias (1 in 12)")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("place of the chosen option in the list (1 = first)")
    ax.set_ylabel("share of choices")
    ax.set_title("Which place in the list Haiku picks", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "positions.png")


def distribution_figure(run, out):
    """Where each group works over the 100 approaches: non-AI users above the line, AI users below."""
    adviser = run.name.split("_theta")[0]
    landscape = int(run.name.split("_L")[1][:2])
    d = pd.read_csv(run / "distributions.csv").sort_values(["t", "x"])
    t_max = d.t.max()
    times = [0, 5, 10, t_max]
    top = 1.1 * d[d.t.isin(times)][["mN", "mA"]].to_numpy().max()  # same scale in every panel
    ticks = np.round(np.linspace(0, top / 1.1, 3), 2)
    fig, axes = plt.subplots(len(times), 1, figsize=(14, 2.3 * len(times) + 0.8), sharex=True)
    for ax, t in zip(axes, times):
        g = d[d.t == t]
        ax.bar(g.x, g.mN, width=0.8, color=INK, label="non-AI users")
        ax.bar(g.x, -g.mA, width=0.8, color=COLORS[adviser], label=f"AI users ({LABELS[adviser]})")
        ax.axhline(0, color=INK, lw=0.6)
        for edge in range(20, 100, 20):
            ax.axvline(edge - 0.5, color=GRID, lw=1)
        ax.set_ylim(-top, top)
        ax.set_yticks([*-ticks[:0:-1], *ticks], [f"{v:g}" for v in [*ticks[:0:-1], *ticks]])
        ax.set_ylabel("share of the group")
        ax.grid(axis="x", visible=False)
        ax.text(0.005, 0.93, f"t = {t}:  non-AI users on {int((g.mN >= 0.01).sum())} approaches,  AI users on "
                f"{int((g.mA >= 0.01).sum())}", transform=ax.transAxes, va="top", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlim(-0.6, 99.6)
    axes[-1].set_xticks([10 * k + 9.5 for k in range(0, 10, 2)], QUESTION_TYPES)
    for ax in axes:
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xlabel("the 100 approaches, grouped by type of research question")
    axes[0].set_title(f"Where researchers work, landscape {landscape}: non-AI users above the line, AI users below",
                      loc="left")
    fig.tight_layout()
    fig.savefig(out / f"{adviser}_L{landscape:02d}.png")
    plt.close(fig)


def main(folder):
    folder = Path(folder)
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    df = measures(folder)
    df.to_csv(folder / "measures.csv", index=False)
    regret_figure(df, out)
    value_figure(df, out)
    approaches_figure(df, out)
    landscape_figure(df, out)
    position_figure(folder, out)
    (out / "distributions").mkdir(exist_ok=True)
    for run in sorted(folder.glob("*_theta*")):
        distribution_figure(run, out / "distributions")
    print(f"figures in {out}")


if __name__ == "__main__":
    main(sys.argv[1])
