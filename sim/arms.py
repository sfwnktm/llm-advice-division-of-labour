"""Compare named arms (a results folder and an adviser in it) on the same landscapes: one figure, one table.

    python -m sim.arms <out.png> "label|folder:adviser" "label|folder:adviser" ...

Per arm and landscape, averaged over periods t = 1..T: AI regret, concentration of the AI users H^A, welfare W;
and at t = T the number of approaches with AI users. Bars: mean over landscapes; dots: single landscapes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .parameters import C
from .plots import INK, measures, plt

PANELS = [("ai_regret", "Regret of AI users (0 = best possible move)"),
          ("W", "Value per researcher, everyone (higher is better)"),
          ("ai_approaches_end", "Approaches used by AI users at the end")]


def arm_table(label, folder, adviser):
    regret = measures(folder)
    regret = regret[regret.adviser == adviser].groupby("landscape").ai_regret.mean()
    rows = []
    for run in sorted(Path(folder).glob(f"{adviser}_theta*")):
        landscape = int(run.name.split("_L")[1][:2])
        d = pd.read_csv(run / "distributions.csv")
        later = d[d.t > 0]
        by_t = later.groupby("t")
        rows.append({"arm": label, "landscape": landscape, "ai_regret": regret[landscape],
                     "H_ai_mean": by_t.apply(lambda g: (g.mA ** 2).sum(), include_groups=False).mean(),
                     "W": by_t.apply(lambda g: (g.m * (g.kappa - C * g.m)).sum(), include_groups=False).mean(),
                     "ai_approaches_end": int((d[d.t == d.t.max()].mA >= 0.01).sum())})
    return pd.DataFrame(rows)


def main(out, specs):
    arms = []
    for spec in specs:
        label, target = spec.split("|", 1)
        folder, adviser = target.rsplit(":", 1)
        arms.append(arm_table(label, folder, adviser))
    df = pd.concat(arms)
    labels = list(dict.fromkeys(df.arm))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.8))
    for ax, (column, title) in zip(axes, PANELS):
        means = df.groupby("arm")[column].mean().reindex(labels)
        if column == "W":  # not a quantity from zero: show the mean as a marker, not a bar
            ax.scatter(means, range(len(labels)), marker="|", s=400, color="#2a78d6", linewidth=3, zorder=2)
        else:
            ax.barh(range(len(labels)), means, color="#86b6ef", height=0.6)
        for i, label in enumerate(labels):
            ax.scatter(df[df.arm == label][column], np.full((df.arm == label).sum(), i), s=12, color=INK, zorder=3)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    lo, hi = df.W.min(), df.W.max()
    axes[1].set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    fig.tight_layout()
    fig.savefig(out)
    print(df.groupby("arm").mean(numeric_only=True).drop(columns="landscape").reindex(labels).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
