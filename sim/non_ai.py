"""Non-AI users: the period equilibrium of spec section 12, solved with MFGLib.

The period is written as an MFGLib game with one move (horizon T = 1):
  state  = the approach x a non-AI user is on now, distributed as m^N_t;
  action = the approach y it moves to;
  reward = -C_t(x, y; m_{t+1}) at the move, 0 afterwards (no terminal cost).
MFGLib's state-action distribution at the move is the flow L^N_t(x, y), so the non-AI crowd next period is its
action marginal, m^N_{t+1}(y) = sum_x L^N_t(x, y). MFGLib cannot forbid actions, so a move outside A(x) costs a
large penalty.

Solver: MFGLib's online mirror descent (Perolat et al., 2022). Each iteration adds the Q-values of the current policy
to a running sum and sets the policy to the softmax of alpha times that sum. The softmax is only the solver's search
step, not a choice rule of the model: the result is accepted when its exploitability, the average gain a non-AI user
could still make by changing its move alone, is below TOL. (MFGLib's fictitious play was tried first; it needs
thousands of iterations here, where mirror descent needs about ten.)
"""
import numpy as np
import torch
from mfglib.alg import OnlineMirrorDescent
from mfglib.env import Environment

from .parameters import C, GAMMA, MAX_ITER, OMD_ALPHA, TOL
from .world import DISTANCE, N

torch.set_default_dtype(torch.float64)  # MFGLib builds its own tensors in the default type; float32 is too coarse

PENALTY = 100.0
MOVE = torch.tensor(GAMMA * DISTANCE + PENALTY * (DISTANCE > 1))  # gamma d(x, y), or the penalty outside A(x)
GOES_TO = torch.eye(N).unsqueeze(1).repeat(1, N, 1)  # P(y' | x, y) = 1[y' = y], shape (y', x, y)


class Reward:
    def __init__(self, kappa_next, mA_next, theta):
        self.kappa_next = torch.tensor(kappa_next)
        self.mA_next = torch.tensor(mA_next)
        self.theta = theta

    def __call__(self, env, t, L_t):
        if t == 1:
            return torch.zeros(N, N)
        mN_next = L_t.sum(dim=0)
        m_next = (1 - self.theta) * mN_next + self.theta * self.mA_next
        cost = MOVE - self.kappa_next[None, :] + C * m_next[None, :]
        return -cost


def period_equilibrium(mN_t, mA_next, kappa_next, theta):
    """Return the non-AI flows L^N_t (N x N), m^N_{t+1}, the exploitability and the number of iterations."""
    env = Environment(
        T=1, S=(N,), A=(N,), mu0=torch.tensor(mN_t),
        r_max=PENALTY + 20.0,
        reward_fn=Reward(kappa_next, mA_next, theta),
        transition_fn=lambda env, t, L_t: GOES_TO,
    )
    policies, exploitabilities, _ = OnlineMirrorDescent(alpha=OMD_ALPHA).solve(env, max_iter=MAX_ITER, atol=TOL, rtol=None)
    best = int(np.argmin(exploitabilities))
    flows = mN_t[:, None] * policies[best][0].numpy()
    return flows, flows.sum(axis=0), exploitabilities[best], best
