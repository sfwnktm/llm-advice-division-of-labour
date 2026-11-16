"""How consistent each effect is over the 5 landscapes, for the report's five measures. No statistical tests.

    python -m sim.consistency

For every comparison of two arms A and B it reports the mean +- SD of each over the 5 landscapes and on how many of
the 5 landscapes A is lower (or higher) than B. Each landscape value is an average over the 15 periods.
Writes results/consistency/consistency.md and consistency.csv. The measures are defined in sim/report_figures.py.
"""
import json
from pathlib import Path

import pandas as pd

from .report_figures import MAIN, SWEEP, by_landscape

OUT = Path("results/consistency")
THETA = {theta: Path("results") / folder for theta, folder in SWEEP.items()}
R5 = {"haiku_thinking": Path("results/R5_haiku_thinking"), "haiku": Path("results/R5_haiku")}
MEMORY_SHOWN = Path("results/memory3_shown_haiku_thinking")
MEMORY_HIDDEN = Path("results/memory3_hidden_haiku_thinking")
NAME = {"haiku_thinking": "Haiku, thinking on", "haiku": "Haiku, thinking off", "naive": "Naive rule"}
LABEL = {"ai_regret": "AI regret", "H": "Concentration H", "breadth": "Breadth", "D": "Dissimilarity D", "W": "Welfare W"}
# The direction each measure is expected to move when advice is followed by more people or herding gets stronger.
WORSE = {"ai_regret": "higher", "H": "higher", "breadth": "lower", "D": "higher", "W": "lower"}
BETTER = {measure: "lower" if direction == "higher" else "higher" for measure, direction in WORSE.items()}


def first_place_share(adviser):
    """Per landscape: how often Haiku picks the option listed first (1 in 12 = 0.083 without bias)."""
    shares = {}
    for run in sorted(MAIN.glob(f"{adviser}_theta*")):
        places = [json.loads(line)["position"] for line in open(run / "calls.jsonl")]
        shares[int(run.name.split("_L")[1][:2])] = sum(p == 1 for p in places) / len(places)
    return pd.Series(shares)


def row(section, measure, a_name, a, b_name, b, direction):
    count = int((a < b).sum() if direction == "lower" else (a > b).sum())
    return {"section": section, "measure": measure,
            "A": a_name, "A mean ± SD": f"{a.mean():.3f} ± {a.std():.3f}",
            "B": b_name, "B mean ± SD": f"{b.mean():.3f} ± {b.std():.3f}",
            "result": f"A {direction} than B on {count} of {len(a)} landscapes"}


def compare(section, a_name, a, b_name, b, directions, measures=tuple(LABEL)):
    """One row per measure: is arm A lower or higher than arm B, landscape by landscape?"""
    return [row(section, LABEL[m], a_name, a[m], b_name, b[m], directions[m]) for m in measures]


def main():
    rows = []
    no_ai = by_landscape(THETA[0.0], "stay")  # theta = 0: nobody follows an adviser

    # 1. More AI users
    for adviser in ["haiku_thinking", "naive", "haiku"]:
        at = {theta: by_landscape(folder, adviser) for theta, folder in THETA.items() if theta > 0}
        section = f"1. More AI users ({NAME[adviser]})"
        rows += compare(section, "θ = 1", at[1.0], "θ = 0.1", at[0.1], WORSE, ["ai_regret"])
        rows += compare(section, "θ = 1", at[1.0], "nobody uses AI (θ = 0)", no_ai, WORSE, ["H", "breadth", "W"])
        rows += compare(section, "θ = 0.75", at[0.75], "θ = 0.1", at[0.1], WORSE, ["D"])
        rows += compare(section, "θ = 0.5", at[0.5], "nobody uses AI (θ = 0)", no_ai, WORSE, ["H", "breadth", "W"])

    # 2. Advisers against each other
    for theta in [0.1, 0.25, 0.5, 0.75, 1.0]:
        on, naive, off = (by_landscape(THETA[theta], a) for a in ["haiku_thinking", "naive", "haiku"])
        rows += compare("2. Advisers: AI regret at every θ", f"Haiku, thinking on, θ = {theta:g}", on,
                        "Naive rule", naive, BETTER, ["ai_regret"])
        rows += compare("2. Advisers: AI regret at every θ", f"Haiku, thinking off, θ = {theta:g}", off,
                        "Haiku, thinking on", on, WORSE, ["ai_regret"])
    on, naive, off = (by_landscape(MAIN, a) for a in ["haiku_thinking", "naive", "haiku"])
    rows += compare("2. Advisers: all measures at θ = 0.5", "Haiku, thinking on", on, "Naive rule", naive, BETTER)
    rows += compare("2. Advisers: all measures at θ = 0.5", "Haiku, thinking off", off, "Haiku, thinking on", on, BETTER)

    # 3. Five answers per group instead of one (theta = 0.5)
    for adviser in ["haiku_thinking", "haiku"]:
        rows += compare("3. Five answers per group (θ = 0.5)", f"{NAME[adviser]}, 5 answers",
                        by_landscape(R5[adviser], adviser), "same, 1 answer", by_landscape(MAIN, adviser), BETTER)

    # 4. Memory of the last 3 studies (Haiku, thinking on, theta = 0.5)
    none, shown, hidden = (by_landscape(f, "haiku_thinking") for f in [MAIN, MEMORY_SHOWN, MEMORY_HIDDEN])
    rows += compare("4. Memory (Haiku, thinking on, θ = 0.5)", "memory shown", shown, "memory not shown", hidden, BETTER)
    rows += compare("4. Memory (Haiku, thinking on, θ = 0.5)", "memory not shown", hidden, "no memory", none, BETTER)

    # 5. Place in the list (theta = 0.5): does Haiku avoid the option listed first?
    for adviser in ["haiku_thinking", "haiku"]:
        share = first_place_share(adviser)
        rows.append(row("5. Place in the list (θ = 0.5)", "share of choices that are the first-listed option",
                        NAME[adviser], share, "no bias (1 in 12)", pd.Series(1 / 12, index=share.index), "lower"))

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "consistency.csv", index=False)
    lines = ["# Consistency over the 5 landscapes", "",
             "No statistical tests. Each value is one landscape's average over the 15 periods; the table gives the "
             "mean ± SD over the 5 landscapes and on how many of them the effect appears.", ""]
    for section, part in table.groupby("section", sort=False):
        lines += [f"## {section}", "", "| measure | A | A mean ± SD | B | B mean ± SD | result |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.measure} | {r.A} | {r['A mean ± SD']} | {r.B} | {r['B mean ± SD']} | {r.result} |"
                  for _, r in part.iterrows()]
        lines.append("")
    (OUT / "consistency.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
