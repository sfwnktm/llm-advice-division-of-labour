"""Approaches, distance, options, significance and depletion (spec sections 2-5 and 8)."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .parameters import KAPPA_MEAN, LAMBDA

# An approach is x = (q, g, a): question type (5), data gathering (5), analysis (4). The groups are those of the
# model paper, in its order; the names are the plainer ones of the earlier prompts (paper name in the comment).
QUESTION_TYPES = ["exploratory", "base-rate", "relationship", "causality", "design"]  # Easterbrook et al. (2008)
DATA_GATHERING = ["self-report",              # asking people
                  "observation and testing",  # observing and testing
                  "instrument measurement",   # physical measurement
                  "secondary data",           # existing material
                  "simulated data"]           # generated data
ANALYSIS = ["statistical analysis",                   # statistical
            "computational analysis",                 # computational
            "qualitative analysis",                   # qualitative
            "mixed, design and conceptual analysis"]  # synthesis, design, conceptual
APPROACHES = list(itertools.product(range(5), range(5), range(4)))
N = len(APPROACHES)  # 100

# d(x, y): the number of parts in which x and y differ.
DISTANCE = np.array([[sum(xp != yp for xp, yp in zip(x, y)) for y in APPROACHES] for x in APPROACHES])

# A(x): the 12 approaches with d(x, y) <= 1, x itself included.
OPTIONS = [np.flatnonzero(DISTANCE[x] <= 1) for x in range(N)]


def draw_kappa0(landscape):
    """kappa_0(x) = 6.7 + (phi_q + phi_g + phi_a) / sqrt(3).

    NK landscape with N = 3 parts and K = 2: each part's contribution depends on its own value and on the two
    other parts, so it is a table over the whole approach, drawn standard normal.
    """
    rng = np.random.default_rng(landscape)
    phi = rng.standard_normal((3, 5, 5, 4))  # phi_q, phi_g, phi_a, each indexed by (q, g, a)
    return np.array([KAPPA_MEAN + (phi[0][x] + phi[1][x] + phi[2][x]) / np.sqrt(3) for x in APPROACHES])


def deplete(kappa_t, m_t):
    """kappa_{t+1}(x) = kappa_t(x) (1 - lambda m_t(x)). No recovery."""
    return kappa_t * (1 - LAMBDA * m_t)


M0_FILE = Path(__file__).resolve().parent.parent / "inputs" / "m0.csv"


def starting_distribution():
    """m_0 from inputs/m0.csv, the same for both populations.

    The file holds only the 100 shares. They were computed once from the survey of 12 researchers, which stays
    outside this repository: each response's research question coded into a question type, its methods mapped to
    the data-gathering and analysis groups (model paper, Appendix B), weight 1 / (k_g k_a) on each of its k_g x k_a
    approaches, then the average over responses.
    """
    m = pd.read_csv(M0_FILE).sort_values("x").m0.to_numpy()
    return m / m.sum()
