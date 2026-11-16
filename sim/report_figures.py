"""Figures for the report, with the report's five measures and no captions on the figures.

    python -m sim.report_figures        # writes results/report_figures_v3/

  AI regret         cost of the AI users' moves minus the cost of their best option, given the crowd that followed
  Concentration H   sum_x m_t(x)^2, from 0.01 (spread evenly) to 1 (all on one approach)
  Breadth           number of approaches with more than 1% of researchers
  Dissimilarity D   0.5 sum_x |m^N_t(x) - m^A_t(x)|, from 0 (same places) to 1 (completely different places)
  Welfare W         sum_x m_t(x) (kappa_t(x) - c m_t(x)), the significance obtained minus crowding

Lines are means over the 5 landscapes, bands +- 1 SD. Each period t = 1..15 shows the crowd m_t and the regret of the
move that led to it.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import COLORS, INK, measures, plt
from .theta import population

OUT = Path("results/report_figures_v3")
SWEEP = {0.0: "2026-11-16_064536", 0.1: "2026-11-16_065022", 0.25: "2026-11-16_065023", 0.5: "2026-11-16_063459",
         0.75: "2026-11-16_065025", 1.0: "2026-11-16_065027"}
MAIN = Path("results/2026-11-16_063459")  # theta = 0.5
MEASURES = [("ai_regret", "AI regret", "0 = best possible move"),
            ("H", "Concentration H", "0.01 = spread evenly, 1 = all on one approach"),
            ("breadth", "Breadth", "approaches with more than 1% of researchers"),
            ("D", "Dissimilarity D of the two groups", "0 = same places, 1 = completely different places"),
            ("W", "Welfare W", "significance obtained minus crowding")]
ORANGE, GREEN, BLUE, GREY = COLORS["haiku_thinking"], COLORS["haiku"], COLORS["naive"], "#8a8985"

# An arm is (label, results folder, adviser, colour, dashed line).
ADVISER_ARMS = [("Naive rule", MAIN, "naive", BLUE, False),
                ("Haiku, thinking on", MAIN, "haiku_thinking", ORANGE, False),
                ("Haiku, thinking off", MAIN, "haiku", GREEN, False)]
MEMORY_ARMS = [("no memory", MAIN, "haiku_thinking", ORANGE, False),
               ("remembers its last 3 studies", Path("results/memory3_shown_haiku_thinking"), "haiku_thinking", BLUE, False),
               ("same groups, memory not shown to Haiku", Path("results/memory3_hidden_haiku_thinking"), "haiku_thinking",
                GREY, True)]
R5_ARMS = [("thinking on, 1 answer per group", MAIN, "haiku_thinking", ORANGE, False),
           ("thinking on, 5 answers per group", Path("results/R5_haiku_thinking"), "haiku_thinking", ORANGE, True),
           ("thinking off, 1 answer per group", MAIN, "haiku", GREEN, False),
           ("thinking off, 5 answers per group", Path("results/R5_haiku"), "haiku", GREEN, True)]
ALL_ARMS = [("Naive rule", MAIN, "naive", BLUE),
            ("Haiku, thinking on", MAIN, "haiku_thinking", ORANGE),
            ("Haiku, thinking on, 5 answers", Path("results/R5_haiku_thinking"), "haiku_thinking", ORANGE),
            ("Haiku, thinking on, memory shown", Path("results/memory3_shown_haiku_thinking"), "haiku_thinking", ORANGE),
            ("Haiku, thinking on, memory not shown", Path("results/memory3_hidden_haiku_thinking"), "haiku_thinking", ORANGE),
            ("Haiku, thinking off", MAIN, "haiku", GREEN),
            ("Haiku, thinking off, 5 answers", Path("results/R5_haiku"), "haiku", GREEN),
            ("Random", MAIN, "random", COLORS["random"]), ("Stay", MAIN, "stay", COLORS["stay"])]


@lru_cache(maxsize=None)
def five(folder):
    """One row per adviser, landscape and period t = 1..T with the five measures."""
    regret = measures(folder)[["adviser", "landscape", "t", "ai_regret"]]
    regret = regret.assign(t=regret.t + 1)  # the move made at t - 1 leads to the crowd m_t
    return population(folder).merge(regret, on=["adviser", "landscape", "t"], how="left")


def by_landscape(folder, adviser):
    """The five measures averaged over the periods, one row per landscape."""
    df = five(folder)
    return df[df.adviser == adviser].groupby("landscape")[[column for column, _, _ in MEASURES]].mean()


def line(ax, x, mean, sd, color, label, dashed=False):
    ax.plot(x, mean, marker="o", color=color, lw=2, ms=4, ls="--" if dashed else "-", label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12, lw=0)


def five_panels(legend_title):
    """Five panels and a sixth cell for the legend. Returns the five axes and a function that saves the figure."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    panels = list(axes.flat[:5])
    for ax, (_, title, ylabel) in zip(panels, MEASURES):
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)

    def finish(name):
        handles, labels = panels[0].get_legend_handles_labels()
        axes[1, 2].axis("off")
        axes[1, 2].legend(handles, labels, loc="center", fontsize=10, title=legend_title, title_fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / name)
        plt.close(fig)

    return panels, finish


def sweep_table():
    """The five measures per adviser, theta and landscape. theta = 0 (no AI users) is the same for every adviser."""
    rows = []
    for theta, folder in SWEEP.items():
        for adviser in ["naive", "haiku_thinking", "haiku"]:
            source = "stay" if theta == 0 else adviser
            rows.append(by_landscape(Path("results") / folder, source).assign(adviser=adviser, theta=theta).reset_index())
    return pd.concat(rows)


def figure_sweep():
    df = sweep_table()
    thetas = sorted(df.theta.unique())
    panels, finish = five_panels("AI users follow")
    for ax, (column, _, _) in zip(panels, MEASURES):
        for label, _, adviser, color, _ in ADVISER_ARMS:
            g = df[df.adviser == adviser].groupby("theta")[column]
            mean, sd = g.mean().dropna(), g.std().dropna()
            line(ax, mean.index, mean, sd.reindex(mean.index), color, label)
        ax.set_xticks(thetas, [f"{t:g}" for t in thetas])
        ax.set_xlim(-0.04, 1.04)
        ax.set_xlabel("share of AI users, θ")
    finish("fig1_theta_sweep.png")


def figure_over_time(arms, legend_title, name):
    panels, finish = five_panels(legend_title)
    for label, folder, adviser, color, dashed in arms:
        df = five(folder)
        df = df[df.adviser == adviser]
        for ax, (column, _, _) in zip(panels, MEASURES):
            g = df.groupby("t")[column]
            line(ax, g.mean().index, g.mean(), g.std(), color, label, dashed)
    for ax in panels:
        ax.set_xticks(range(1, 16))
        ax.set_xlabel("t (period)")
    finish(name)


def figure_all_arms(arms, name):
    """Arms at theta = 0.5: one dot per landscape, a bar mark at the mean."""
    labels = [label for label, _, _, _ in arms]
    fig, axes = plt.subplots(1, 5, figsize=(18, 0.4 * len(arms) + 0.9), sharey=True)
    for ax, (column, title, xlabel) in zip(axes, MEASURES):
        for i, (label, folder, adviser, color) in enumerate(arms):
            values = by_landscape(folder, adviser)[column]
            ax.scatter(values, np.full(len(values), i), s=14, color=color, alpha=0.5)
            ax.scatter(values.mean(), i, s=90, marker="|", color=color, linewidth=3)
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def figure_agreement():
    """When Haiku is asked 5 times about the same group, how many different answers does it give?"""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, (label, folder, adviser) in enumerate([("Haiku, thinking on", "results/R5_haiku_thinking", "haiku_thinking"),
                                                  ("Haiku, thinking off", "results/R5_haiku", "haiku")]):
        answers = {}
        for run in sorted(Path(folder).glob(f"{adviser}_theta*")):
            for call in pd.read_json(run / "calls.jsonl", lines=True).to_dict("records"):
                answers.setdefault((run.name, call["t"], call["x"]), set()).add(call["y"])
        different = np.array([len(v) for v in answers.values()])
        share = [(different == n).mean() for n in range(1, 6)]
        ax.bar(np.arange(1, 6) + (k - 0.5) * 0.4, share, width=0.38, color=COLORS[adviser], label=label)
        for n, v in enumerate(share, start=1):
            ax.text(n + (k - 0.5) * 0.4, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(1, 6), ["1\n(all agree)", "2", "3", "4", "5\n(all differ)"])
    ax.set_xlabel("different answers among the 5")
    ax.set_ylabel("share of groups")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Asked 5 times about the same group, how many different answers does Haiku give?", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_five_answers_agreement.png")
    plt.close(fig)


def forecast_errors():
    """One row per call: error of the forecast and of "nothing changes" on the group's 12 options, in percentage points.

    Haiku gives a forecast. The Naive rule gives none: it uses today's shares, which is the "nothing changes" forecast.
    """
    from .world import OPTIONS
    rows = []
    for adviser in ["haiku_thinking", "haiku", "naive"]:
        for run in sorted(MAIN.glob(f"{adviser}_theta*")):
            d = pd.read_csv(run / "distributions.csv").sort_values(["t", "x"])
            shares = {t: g.m.to_numpy() for t, g in d.groupby("t")}
            for call in pd.read_json(run / "calls.jsonl", lines=True).to_dict("records"):
                forecast = call.get("forecast")
                ys = np.array([int(y) for y in forecast]) if isinstance(forecast, dict) else OPTIONS[call["x"]]
                real = shares[call["t"] + 1][ys]
                rows.append({"adviser": adviser, "landscape": int(run.name.split("_L")[1][:2]),
                             "forecast": 100 * np.abs(np.array(list(forecast.values()), float) - real).mean()
                             if isinstance(forecast, dict) else np.nan,
                             "nothing changes": 100 * np.abs(shares[call["t"]][ys] - real).mean()})
    return pd.DataFrame(rows)


def figure_forecast():
    """Forecast error of Haiku and of the Naive rule, against simply assuming nothing changes."""
    df = forecast_errors().groupby("adviser")[["forecast", "nothing changes"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    light = "#d6d5cf"
    bars = [(0, "haiku_thinking", "forecast", ORANGE), (1, "haiku_thinking", "nothing changes", light),
            (2.4, "haiku", "forecast", GREEN), (3.4, "haiku", "nothing changes", light),
            (4.8, "naive", "nothing changes", BLUE)]
    for x, adviser, column, color in bars:
        value = df.loc[adviser, column]
        ax.bar(x, value, width=0.9, color=color)
        ax.text(x, value + 0.04, f"{value:.2f}", ha="center", fontsize=10, color=INK)
    for x, name, color in [(0.5, "Haiku, thinking on", ORANGE), (2.9, "Haiku, thinking off", GREEN),
                           (4.8, "Naive rule", BLUE)]:
        ax.text(x, -0.52, name, ha="center", fontsize=10, fontweight="bold", color=color)
    ax.set_xticks([x for x, *_ in bars], ["Haiku's\nforecast", "assuming\nnothing changes"] * 2
                  + ["uses today's shares\n(= nothing changes)"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.15 * df.max().max())
    ax.set_ylabel("average error (percentage points)")
    ax.set_title("How wrong is the forecast of where researchers go next?", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT / "fig7_forecast.png")
    plt.close(fig)
    return df


def figure_forecast_haiku():
    """Haiku's forecast of next period's shares against today's shares: bars over all calls, dots per landscape."""
    calls = forecast_errors()
    overall = calls.groupby("adviser")[["forecast", "nothing changes"]].mean()
    per_landscape = calls.groupby(["adviser", "landscape"])[["forecast", "nothing changes"]].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    light = "#d6d5cf"
    for i, (adviser, name, color) in enumerate([("haiku_thinking", "Haiku, thinking on", ORANGE),
                                                ("haiku", "Haiku, thinking off", GREEN)]):
        left, right = 2.4 * i, 2.4 * i + 1
        for x, column, fill in [(left, "forecast", color), (right, "nothing changes", light)]:
            ax.bar(x, overall.loc[adviser, column], width=0.9, color=fill)
            ax.text(x, 0.12, f"{overall.loc[adviser, column]:.2f}", ha="center", fontsize=12, fontweight="bold",
                    color=INK if fill == light else "white")
        for _, r in per_landscape.loc[adviser].iterrows():  # the same landscape joined by a line
            ax.plot([left, right], [r["forecast"], r["nothing changes"]], color=INK, lw=0.8, marker="o", ms=3, alpha=0.6)
        ax.text((left + right) / 2, -0.58, name, ha="center", fontsize=10, fontweight="bold", color=color)
    ax.set_xticks([0, 1, 2.4, 3.4], ["Haiku's forecast\nof $m_{t+1}$", "today's shares\n$m_t$"] * 2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("average error (percentage points)")
    ax.set_title("Error in predicting next period's shares $m_{t+1}$", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT / "fig8_forecast_haiku.png")
    plt.close(fig)
    worse = (per_landscape["forecast"] > per_landscape["nothing changes"]).groupby("adviser").sum()
    return overall, per_landscape, worse


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_sweep()
    figure_over_time(ADVISER_ARMS, "AI users follow (θ = 0.5)", "fig2_over_time.png")
    figure_all_arms(ALL_ARMS, "fig3_all_arms.png")
    figure_all_arms([arm for arm in ALL_ARMS if "memory" not in arm[0]], "fig3_all_arms_without_memory.png")
    figure_over_time(MEMORY_ARMS, "Haiku, thinking on (θ = 0.5)", "fig4_memory.png")
    figure_over_time(R5_ARMS, "Haiku (θ = 0.5)", "fig5_five_answers.png")
    figure_agreement()
    figure_forecast()
    figure_forecast_haiku()
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
