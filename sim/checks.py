"""Checks of the implementation against the spec. No API calls.

    python -m sim.checks
"""
import json

import numpy as np

from .advisers import Call, naive, naive_choice, random, stay
from .llm import KEYS, build_prompt, describe, parse
from .non_ai import period_equilibrium
from .parameters import C, GAMMA
from .simulate import simulate
from .world import APPROACHES, DISTANCE, N, OPTIONS, QUESTION_TYPES, deplete, draw_kappa0, starting_distribution


def check(name, condition):
    print(f"{'ok  ' if condition else 'FAIL'} {name}")
    assert condition, name


def world():
    check("100 approaches, 12 options each, x among its options",
          N == 100 and all(len(o) == 12 and x in o for x, o in enumerate(OPTIONS)))
    check("d is symmetric, 0 on the diagonal, in {0, 1, 2, 3}",
          (DISTANCE == DISTANCE.T).all() and (np.diag(DISTANCE) == 0).all() and set(np.unique(DISTANCE)) == {0, 1, 2, 3})
    kappa = np.concatenate([draw_kappa0(ell) for ell in range(200)])
    check(f"kappa_0 ~ N(6.7, 1): mean {kappa.mean():.3f}, SD {kappa.std():.3f}",
          abs(kappa.mean() - 6.7) < 0.02 and abs(kappa.std() - 1) < 0.02)
    check("landscapes are reproducible and differ", (draw_kappa0(3) == draw_kappa0(3)).all()
          and not (draw_kappa0(3) == draw_kappa0(4)).all())
    m = starting_distribution()
    check("depletion: kappa_{t+1} = kappa_t (1 - m_t)", np.allclose(deplete(draw_kappa0(0), m), draw_kappa0(0) * (1 - m)))
    by_question = [m[[x for x in range(N) if APPROACHES[x][0] == q]].sum() for q in range(5)]
    check(f"m_0 from the survey: adds up to 1, {int((m > 0).sum())} approaches occupied, by question type "
          f"{np.round(by_question, 3).tolist()} (6 design, 3 exploratory, 3 causality of 12)",
          np.isclose(m.sum(), 1) and (m >= 0).all() and np.allclose(by_question, [0.25, 0, 0, 0.25, 0.5]))


def non_ai_step():
    rng = np.random.default_rng(1)
    for theta in (0.0, 0.5, 0.9):
        mN = rng.dirichlet(np.full(N, 0.1))
        mA_next = rng.dirichlet(np.full(N, 0.1))
        kappa_next = deplete(draw_kappa0(2), (1 - theta) * mN + theta * mA_next)
        flows, mN_next, exploitability, _ = period_equilibrium(mN, mA_next, kappa_next, theta)
        cost = GAMMA * DISTANCE - kappa_next[None] + C * ((1 - theta) * mN_next + theta * mA_next)[None]
        gap = np.array([cost[x] - cost[x, OPTIONS[x]].min() for x in range(N)])
        check(f"theta {theta}: flows leave x with mass m^N_t(x) and add up to m^N_(t+1)",
              np.allclose(flows.sum(1), mN) and np.allclose(flows.sum(0), mN_next))
        check(f"theta {theta}: no flow outside A(x)", flows[DISTANCE > 1].sum() < 1e-12)
        check(f"theta {theta}: exploitability {exploitability:.1e}, mass on moves worse than the best by "
              f"> 0.01: {flows[gap > 0.01].sum():.1e}", exploitability < 1e-4 and flows[gap > 0.01].sum() < 0.01)


def advisers_and_prompt():
    rng = np.random.default_rng(2)
    m_now, m_prev = rng.dirichlet(np.ones(N)), rng.dirichlet(np.ones(N))
    calls = [Call(t=3, x=x, seed=0, kappa_next=draw_kappa0(0), m_now=m_now, m_prev=m_prev) for x in range(N)]
    check("Stay returns x", [a["y"] for a in stay(calls)] == list(range(N)))
    check("Random stays inside A(x)", all(a["y"] in OPTIONS[c.x] for c, a in zip(calls, random(calls))))
    by_hand = [min(OPTIONS[c.x], key=lambda y: GAMMA * DISTANCE[c.x, y] - c.kappa_next[y] + C * m_now[y]) for c in calls]
    check("Naive is the minimiser of gamma d - kappa_{t+1} + c m_t", [a["y"] for a in naive(calls)] == by_hand)
    check("the option order is the same for the same (seed, t, x) and differs for another seed",
          (calls[5].shown() == calls[5].shown()).all()
          and not (calls[5].shown() == Call(3, 5, 1, calls[5].kappa_next, m_now, m_prev).shown()).all())
    call = calls[7]
    shown = call.shown()
    prompt = build_prompt(call, shown)
    check("the prompt lists the 12 options with names, d, kappa_{t+1}, m_t and m_{t-1}",
          all(f"{i}. {describe(y)}; parts changed: {DISTANCE[call.x, y]}; significance left: "
              f"{call.kappa_next[y]:.3f}; share of all researchers now: {m_now[y]:.4f}; "
              f"share of all researchers last period: {m_prev[y]:.4f}" in prompt for i, y in enumerate(shown, 1)))
    first = Call(0, 7, 0, call.kappa_next, m_now, None)
    first_prompt = build_prompt(first, first.shown())
    check("at t = 0 the prompt has no last-period share", "last period" not in first_prompt)
    check("the prompt names the group's current approach and states the value rule",
          f"- Type of research question: {QUESTION_TYPES[APPROACHES[7][0]]}" in prompt
          and f"significance left - {C} x share of all researchers on that approach next period - {GAMMA} x parts "
              f"changed" in prompt)
    answer = {"forecast": {k: 0.01 * i for i, k in enumerate(KEYS)}, "option": 4}
    position, forecast = parse({"content": [{"type": "text", "text": json.dumps(answer)}]})
    check("an answer maps back to the approach shown at that position", shown[position - 1] == shown[3] and len(forecast) == 12)
    check("an answer out of range is invalid", parse({"content": [{"type": "text", "text": '{"option": 13}'}]}) is None)


def runs():
    for adviser in ("stay", "random", "naive"):
        for theta in (0.0, 0.5, 1.0):
            dist, calls, periods = simulate(adviser, theta, landscape=0, seed=0, periods=3)
            shares = dist.groupby("t")[["mN", "mA", "m"]].sum()
            kappa = dist.pivot(index="t", columns="x", values="kappa").to_numpy()
            check(f"{adviser}, theta {theta}: shares add up to 1, kappa never rises",
                  np.allclose(shares, 1) and (np.diff(kappa, axis=0) <= 1e-12).all())
    dist, calls, _ = simulate("stay", 0.5, landscape=0, seed=0, periods=3)
    mA = dist.pivot(index="t", columns="x", values="mA").to_numpy()
    check("with Stay the AI users never move", np.allclose(mA, mA[0]))
    dist, calls, _ = simulate("naive", 0.5, landscape=0, seed=0, periods=3)
    check("the match rate of Naive with itself is 1", all(c["y"] == c["y_naive"] for c in calls))
    R = 5
    dist, calls, _ = simulate("random", 0.5, landscape=0, seed=0, periods=1, answers=R)
    per_x = {}
    for c in calls:
        per_x.setdefault(c["x"], []).append(c)
    mA1 = dist[dist.t == 1].sort_values("x").mA.to_numpy()
    rebuilt = np.zeros(N)
    for x, cs in per_x.items():
        for c in cs:
            rebuilt[c["y"]] += c["mass"]
    check(f"R = {R}: every occupied approach gets R calls, each moving 1/R of its group",
          all(len(cs) == R and len({c["r"] for c in cs}) == R for cs in per_x.values())
          and all(abs(c["mass"] - starting_distribution()[c["x"]] / R) < 1e-15 for c in calls)
          and np.allclose(rebuilt, mA1))
    plain, _, _ = simulate("naive", 0.5, landscape=1, seed=0, periods=6)
    remembering, calls, periods = simulate("naive", 0.5, landscape=1, seed=0, periods=6, memory=3)
    check("memory k = 3 with Naive (which ignores it): the same crowd path as k = 0, more groups",
          np.allclose(plain[["mN", "mA", "m"]], remembering[["mN", "mA", "m"]])
          and periods.ai_groups.max() >= periods.ai_calls.min())
    check("memory: histories hold at most k = 3 studies; the group shares add up to 1",
          max(c["history_length"] for c in calls) == 3
          and np.allclose(remembering.groupby("t").mA.sum(), 1))
    rng = np.random.default_rng(3)
    m_now = rng.dirichlet(np.ones(N))
    history = ((5, 7.1, 0.02), (6, 6.9, 0.04), (7, 6.5, 0.05))
    shown = Call(4, 7, 0, draw_kappa0(0), m_now, m_now, 0, history, True)
    hidden = Call(4, 7, 0, draw_kappa0(0), m_now, m_now, 0, history, False)
    check("the prompt lists the previous studies only when shown",
          "previous studies" in build_prompt(shown, shown.shown()) and "significance found: 7.100" in
          build_prompt(shown, shown.shown()) and "previous studies" not in build_prompt(hidden, hidden.shown()))


if __name__ == "__main__":
    world()
    non_ai_step()
    advisers_and_prompt()
    runs()
    print("all checks passed")
