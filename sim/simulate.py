"""One run of the model (spec section 14): one adviser, one theta, one landscape, one seed.

AI users are tracked as groups. A group's state is (x, history): its approach and, with memory k > 0, its last k
studies (approach, significance it found there, share of all researchers there). Groups with the same state are one
group. With k = 0 the history is always empty, so a group is just an approach, as in the spec. Memory is an extension
of the spec, whose section 13 has no memory.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import advisers
from .advisers import Call, naive_choice
from .llm import language_model_adviser
from .non_ai import period_equilibrium
from .parameters import MEMORY, R, T
from .world import APPROACHES, N, deplete, draw_kappa0, starting_distribution

ADVISERS = {"stay": advisers.stay, "random": advisers.random, "naive": advisers.naive,
            **{name: language_model_adviser(name) for name in ("haiku", "haiku_thinking")}}


def simulate(adviser, theta, landscape, seed, periods=T, answers=R, memory=MEMORY, show_history=True):
    """Return the distributions over time, one record per AI call, and one record per period.

    answers: R, the answers per group and period; each answer takes 1/R of the group.
    memory: k, the number of past studies a group remembers; show_history: whether the prompt shows them.
    """
    kappa = draw_kappa0(landscape)
    mN = starting_distribution()
    groups = {(int(x), ()): share for x, share in enumerate(starting_distribution()) if share > 0}  # AI users
    m_prev = None
    distributions, calls_log, periods_log = [], [], []

    for t in tqdm(range(periods), desc=f" {adviser}, periods", unit="period", leave=False):
        mA = shares_by_approach(groups)
        # 1. The crowd m_t is given.
        m = (1 - theta) * mN + theta * mA
        # 2. Work: every user on x obtains kappa_t(x) (recorded with the distributions).
        distributions.append(snapshot(t, kappa, mN, mA, m))
        # 3. Depletion.
        kappa_next = deplete(kappa, m)

        # 4. AI users: R calls per group; each answer takes 1/R of the group, so pi^A_t(y | x, history) is the
        #    share of the R answers that are y. The group then remembers the study it just did.
        calls = [Call(t, x, seed, kappa_next, m, m_prev, r, history, show_history)
                 for (x, history) in groups for r in range(answers)] if theta > 0 else []
        new_groups = defaultdict(float)
        for call, answer in zip(calls, ADVISERS[adviser](calls)):
            mass = groups[(call.x, call.history)] / answers
            study = (call.x, float(kappa[call.x]), float(m[call.x]))
            history = (call.history + (study,))[-memory:] if memory else ()
            new_groups[(answer["y"], history)] += mass
            calls_log.append({"t": t, "x": call.x, "r": call.r, "mass": mass, "history_length": len(call.history),
                              "y_naive": naive_choice(call), **answer})
        if calls:
            groups = dict(new_groups)
        mA_next = shares_by_approach(groups)

        # 5. Non-AI users: the period equilibrium, given where the AI users go.
        if theta < 1:
            _, mN_next, exploitability, iterations = period_equilibrium(mN, mA_next, kappa_next, theta)
        else:
            mN_next, exploitability, iterations = mN, 0.0, 0
        periods_log.append({"t": t, "exploitability_non_ai": exploitability, "solver_iterations": iterations,
                            "ai_calls": len(calls), "ai_groups": len(groups)})

        # 6. The new crowd follows from m^N_{t+1} and m^A_{t+1} at the start of the next period.
        m_prev, kappa, mN = m, kappa_next, mN_next

    mA = shares_by_approach(groups)
    distributions.append(snapshot(periods, kappa, mN, mA, (1 - theta) * mN + theta * mA))
    return pd.concat(distributions, ignore_index=True), calls_log, pd.DataFrame(periods_log)


def shares_by_approach(groups):
    """m^A(x): the AI users' share on each approach, summed over histories."""
    mA = np.zeros(N)
    for (x, _), share in groups.items():
        mA[x] += share
    return mA


def snapshot(t, kappa, mN, mA, m):
    q, g, a = np.array(APPROACHES).T
    return pd.DataFrame({"t": t, "x": np.arange(N), "q": q, "g": g, "a": a,
                         "kappa": kappa, "mN": mN, "mA": mA, "m": m})
