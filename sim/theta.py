"""The theta sweep: population measures as the share of AI users grows.

    python -m sim.theta results/<folder> results/<folder> ...      # one folder per value of theta

Writes results/theta_sweep_<date>/: theta_sweep.csv (one row per adviser, theta, landscape) and theta_sweep.png.

Per run, each measure is averaged over periods t = 1..T of the crowd m_t = (1 - theta) m^N_t + theta m^A_t:
  H        concentration, sum_x m_t(x)^2 (0.01 if spread evenly over 100 approaches, 1 if all on one)
  breadth  number of approaches with m_t(x) > 0.01
  D        dissimilarity of AI and non-AI users, 0.5 sum_x |m^N_t(x) - m^A_t(x)| (only for 0 < theta < 1)
  W        welfare, sum_x m_t(x) (kappa_t(x) - c m_t(x)): significance obtained minus crowding
  AI regret and non-AI cost as in sim/plots.py (cost of the move made, with the crowd that actually followed).
theta = 0 has no AI users, so it is the same point for every adviser.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .parameters import C
from .plots import COLORS, LABELS, measures, present, plt


def population(folder):
    rows = []
    for run in sorted(Path(folder).glob("*_theta*")):
        adviser = run.name.split("_theta")[0]
        theta = float(run.name.split("_theta")[1][:4])
        landscape = int(run.name.split("_L")[1][:2])
        d = pd.read_csv(run / "distributions.csv")
        for t, g in d[d.t > 0].groupby("t"):
            m, mN, mA, kappa = g.m.to_numpy(), g.mN.to_numpy(), g.mA.to_numpy(), g.kappa.to_numpy()
            rows.append({"adviser": adviser, "theta": theta, "landscape": landscape, "t": t,
                         "H": (m ** 2).sum(), "breadth": int((m > 0.01).sum()),
                         "D": 0.5 * np.abs(mN - mA).sum() if 0 < theta < 1 else np.nan,
                         "W": (m * (kappa - C * m)).sum()})
    return pd.DataFrame(rows)


def collect(folders):
    per_run = []
    for folder in folders:
        pop = population(folder).groupby(["adviser", "theta", "landscape"]).mean(numeric_only=True).drop(columns="t")
        theta = pop.index.get_level_values("theta")[0]
        choice = measures(folder).groupby(["adviser", "landscape"])[["ai_regret", "non_ai_cost"]].mean()
        choice["theta"] = theta
        choice = choice.set_index("theta", append=True).reorder_levels(["adviser", "theta", "landscape"])
        pop = pop.join(choice)
        if theta == 1:
            pop["non_ai_cost"] = np.nan  # there are no non-AI users
        per_run.append(pop)
    df = pd.concat(per_run).reset_index()
    zero = df[df.theta == 0]  # no AI users: the same point for every adviser
    df = pd.concat([df[df.theta > 0]] + [zero.assign(adviser=a) for a in present(df[df.theta > 0])],
                   ignore_index=True)
    return df.sort_values(["adviser", "theta", "landscape"])


PANELS = [("ai_regret", "Regret of AI users", "regret (0 = best possible move)"),
          ("breadth", "Approaches in use", "approaches with at least 1% of researchers"),
          ("W", "Value per researcher, everyone", "value per researcher (higher is better)")]


def figure(df, out):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4))
    thetas = sorted(df.theta.unique())
    for ax, (column, title, ylabel) in zip(axes, PANELS):
        for adviser in present(df):
            g = df[df.adviser == adviser].groupby("theta")[column]
            mean, sd = g.mean().dropna(), g.std().dropna()
            ax.plot(mean.index, mean, marker="o", color=COLORS[adviser], lw=2, ms=4, label=LABELS[adviser])
            ax.fill_between(mean.index, mean - sd.reindex(mean.index), mean + sd.reindex(mean.index),
                            color=COLORS[adviser], alpha=0.12, lw=0)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        shown = [t for t in thetas if t > 0] if column == "ai_regret" else thetas  # no AI users at theta = 0
        ax.set_xticks(shown, [f"{t:g}" for t in shown])
        ax.set_xlabel("share of researchers following the adviser, θ")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "theta_sweep.png")


def main(folders):
    out = Path("results") / f"theta_sweep_{datetime.now():%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True)
    df = collect(folders)
    df.to_csv(out / "theta_sweep.csv", index=False)
    (out / "sources.json").write_text(json.dumps(folders, indent=2))
    figure(df, out)
    table = df.groupby(["adviser", "theta"])[[c for c, _, _ in PANELS]].mean().round(3)
    print(table.to_string())
    print(f"saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
