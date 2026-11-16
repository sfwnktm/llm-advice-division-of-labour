"""Every number of the model in one place. Section numbers refer to spec.md."""

# Time (section 1)
T = 15

# Significance and depletion (sections 5 and 8)
KAPPA_MEAN = 6.7
LAMBDA = 1.0

# Cost of a move (section 11)
GAMMA = 0.25
C = 2.0

# Design (section 15). The grid of theta is not fixed by the spec yet.
THETAS = [0.5]
LANDSCAPES = list(range(5))
SEEDS = [0]          # one seed: no repeated runs per landscape
ADVISERS = ["haiku", "haiku_thinking", "stay", "random", "naive"]

# Non-AI users: MFGLib's online mirror descent (section 12)
OMD_ALPHA = 10.0    # step size of mirror descent
TOL = 1e-4          # stop when the exploitability is below this
MAX_ITER = 200

# AI users (section 13): answers per approach and period. The group on x splits evenly over its R answers.
R = 1                # as in the spec
MEMORY = 0           # k, past studies a group remembers; the spec has no memory (section 13)

# Language models (section 13)
MODELS = {"haiku": "claude-haiku-5-5"}
THINKING_OFF = {"claude-haiku-5-5": "disabled"}  # how the model is asked not to think
EFFORT = "low"
MAX_TOKENS = 1000            # adviser "haiku": thinking off
MAX_TOKENS_THINKING = 16000  # adviser "haiku_thinking": thinking on (adaptive), same prompt
MAX_ATTEMPTS = 10   # re-ask an invalid answer at most this often, with a new option order
MAX_PARALLEL = 32
PROMPT_VERSION = "3"

RESULTS_DIR = "results"
