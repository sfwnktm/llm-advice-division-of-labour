"""Run the design of spec section 15: every adviser on the same landscapes, seeds and values of theta.

    python -m sim.run                                   # the full design in parameters.py
    python -m sim.run --advisers stay naive --landscapes 0 --seeds 0 --periods 3     # a small test

Each run is saved in results/<date_time>/<adviser>_theta<theta>_L<landscape>_s<seed>/:
    distributions.csv   kappa_t, m^N_t, m^A_t and m_t for every approach and period
    periods.csv         per period: exploitability of the non-AI step, solver iterations, number of AI calls
    calls.jsonl         one line per AI call; for Haiku with every prompt and full response
    summary.json        counts, the match rate a, position shares, tokens, model and date
"""
import argparse
import json
import resource
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import parameters as P
from .simulate import simulate


def summarise(adviser, calls, periods, elapsed):
    """The counts of section 15: invalid answers, re-asks, the match rate a with Naive, and position shares."""
    attempts = [a for c in calls for a in c.get("attempts", [])]  # only Haiku makes attempts
    usage = [a["response"].get("usage", {}) for a in attempts]
    positions = [c["position"] for c in calls if c.get("position")]
    return {
        "adviser": adviser, "calls": len(calls),
        "invalid_answers": sum(not a["valid"] for a in attempts),
        "calls_without_valid_answer": sum(not c.get("valid", True) for c in calls),
        "match_rate_with_naive": float(np.mean([c["y"] == c["y_naive"] for c in calls])) if calls else None,
        "position_shares": {p: positions.count(p) / len(positions) for p in range(1, 13)} if positions else None,
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "max_exploitability_non_ai": float(periods.exploitability_non_ai.max()),
        "model": P.MODELS.get(adviser.split("_")[0]),
        "thinking": ("adaptive" if adviser.endswith("_thinking") else P.THINKING_OFF[P.MODELS[adviser.split("_")[0]]])
                    if adviser.split("_")[0] in P.MODELS else None, "effort": P.EFFORT,
        "prompt_version": P.PROMPT_VERSION,
        "date": datetime.now().isoformat(timespec="seconds"), "seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advisers", nargs="+", default=P.ADVISERS)
    parser.add_argument("--thetas", nargs="+", type=float, default=P.THETAS)
    parser.add_argument("--landscapes", nargs="+", type=int, default=P.LANDSCAPES)
    parser.add_argument("--seeds", nargs="+", type=int, default=P.SEEDS)
    parser.add_argument("--periods", type=int, default=P.T)
    parser.add_argument("--name", help="name of the results folder (default: date and time)")
    parser.add_argument("--R", type=int, default=P.R, help="answers per group and period")
    parser.add_argument("--memory", type=int, default=P.MEMORY, help="k, past studies a group remembers")
    parser.add_argument("--hide-history", action="store_true", help="keep the memory but do not show it in the prompt")
    args = parser.parse_args()

    out = Path(P.RESULTS_DIR) / (args.name or datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    out.mkdir(parents=True)
    settings = {k: v for k, v in vars(P).items() if k.isupper()} | vars(args)
    (out / "parameters.json").write_text(json.dumps(settings, indent=2))

    started = time.time()
    design = [(th, ell, s, adv) for th in args.thetas for ell in args.landscapes for s in args.seeds
              for adv in args.advisers]
    for theta, landscape, seed, adviser in tqdm(design, desc="runs", unit="run"):
        start = time.time()
        distributions, calls, periods = simulate(adviser, theta, landscape, seed, args.periods, args.R, args.memory,
                                                 not args.hide_history)
        folder = out / f"{adviser}_theta{theta:.2f}_L{landscape:02d}_s{seed}"
        folder.mkdir()
        distributions.to_csv(folder / "distributions.csv", index=False)
        periods.to_csv(folder / "periods.csv", index=False)
        with open(folder / "calls.jsonl", "w") as f:
            for call in calls:
                f.write(json.dumps(call) + "\n")
        summary = summarise(adviser, calls, periods, time.time() - start)
        summary |= {"R": args.R, "memory": args.memory, "history_shown": args.memory > 0 and not args.hide_history}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    minutes = (time.time() - started) / 60
    print(f"saved to {out}; total time {minutes:.1f} min; peak memory {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
