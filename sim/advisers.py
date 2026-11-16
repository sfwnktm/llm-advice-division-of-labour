"""The advisers of spec section 15. An adviser takes the calls of one period and returns one choice y_t(x) per call."""
from dataclasses import dataclass

import numpy as np

from .parameters import C, GAMMA
from .world import DISTANCE, OPTIONS

# Separate random streams, so that each random draw is the same in every arm.
ORDER, RANDOM_CHOICE = 0, 1


@dataclass
class Call:
    """One group of AI users on approach x asks for advice at time t."""
    t: int
    x: int
    seed: int
    kappa_next: np.ndarray       # kappa_{t+1}, all approaches
    m_now: np.ndarray            # m_t
    m_prev: np.ndarray | None    # m_{t-1}; None at t = 0
    r: int = 0                   # which of the R answers for this approach (each has its own option order)
    history: tuple = ()          # the group's last k studies: (approach, significance found, share there)
    show_history: bool = True    # whether the prompt shows the history

    def rng(self, stream, attempt=0):
        return np.random.default_rng([self.seed, self.t, self.x, self.r, stream, attempt])

    def shown(self, attempt=0):
        """The 12 options in the order shown to the adviser: random, and the same in every arm."""
        return self.rng(ORDER, attempt).permutation(OPTIONS[self.x])


def naive_choice(call):
    """The minimiser over y in A(x) of gamma d(x, y) - kappa_{t+1}(y) + c m_t(y)."""
    ys = OPTIONS[call.x]
    cost = GAMMA * DISTANCE[call.x, ys] - call.kappa_next[ys] + C * call.m_now[ys]
    return int(ys[np.argmin(cost)])


def stay(calls):
    return [{"y": call.x} for call in calls]


def random(calls):
    return [{"y": int(call.rng(RANDOM_CHOICE).choice(OPTIONS[call.x]))} for call in calls]


def naive(calls):
    return [{"y": naive_choice(call)} for call in calls]
