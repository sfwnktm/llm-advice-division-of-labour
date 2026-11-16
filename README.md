# LLM advice and the division of research labour

A simulation of what happens to a research field when a share of its researchers follows the advice of the same
language model. Researchers divide their work over 100 research approaches. Work uses up the significance of an
approach, and crowding lowers its value. A share θ of researchers (AI users) follows an adviser; the others (non-AI
users) choose at the period equilibrium. The model is specified in [`spec.md`](spec.md); the code refers to its
section numbers.

## Install

Python 3.11.

```bash
pip install -r requirements.txt
```

## Run without a language model

These need no API access and take seconds.

```bash
python -m sim.checks                                                        # checks of the code against spec.md
python -m sim.run --advisers stay random naive --landscapes 0 --name demo   # one landscape, three advisers
python -m sim.plots results/demo                                            # figures in results/demo/figures/
```

## Run with Claude Haiku

The advisers `haiku` (thinking off) and `haiku_thinking` (thinking on) call Claude Haiku 5.5 through Anthropic's
`ant` command-line client, which must be installed and set up with an API key. The code never reads or stores the key.

```bash
python -m sim.run                                             # the design in sim/parameters.py: 5 advisers, 5 landscapes
python -m sim.run --thetas 0.1                                # another share of AI users
python -m sim.run --advisers haiku --R 5                      # 5 answers per group instead of 1
python -m sim.run --advisers haiku_thinking --memory 3        # groups remember their last 3 studies
```

Each run is saved in `results/<name>/<adviser>_theta<θ>_L<landscape>_s<seed>/`:

| File | Content |
|---|---|
| `distributions.csv` | significance and the shares of non-AI users, AI users and everyone, for every approach and period |
| `periods.csv` | per period: exploitability of the non-AI step, solver iterations, number of calls |
| `calls.jsonl` | one line per call to the adviser; for Haiku with every prompt and the full response |
| `summary.json` | counts, agreement with the Naive rule, tokens, model, settings, date |

## Advisers

| Name | Choice for a group on approach x |
|---|---|
| `haiku`, `haiku_thinking` | the answer of Claude Haiku 5.5 to the prompt of spec section 13 |
| `naive` | the cheapest option under the cost rule, with today's shares in place of next period's |
| `stay` | stay on x |
| `random` | one of the 12 options at random |

## Measures

- **AI regret**: the cost of the AI users' moves minus the cost of their best option, given the crowd that followed
  (0 is the best possible move).
- **Concentration** H = Σ m(x)², from 0.01 (spread evenly) to 1 (all on one approach).
- **Breadth**: the number of approaches with more than 1% of researchers.
- **Dissimilarity** of the two groups, D = ½ Σ |mᴺ(x) − mᴬ(x)|, from 0 (same places) to 1 (completely different
  places); the Duncan segregation index.
- **Welfare** W = Σ m(x) (κ(x) − c m(x)): the significance obtained minus crowding.

## Files

| File | Spec | What it does |
|---|---|---|
| `sim/parameters.py` | all | every number in one place |
| `sim/world.py` | 2–5, 8 | approaches, distance, options, significance landscape, depletion, starting distribution |
| `sim/non_ai.py` | 12 | period equilibrium of the non-AI users, as a one-move game in MFGLib |
| `sim/advisers.py` | 15 | Stay, Random, Naive, and the option order of each call |
| `sim/llm.py` | 13 | Haiku as adviser: the prompt, the call, the answer |
| `sim/simulate.py` | 14 | one run |
| `sim/run.py` | 15 | the design: all advisers on the same landscapes and θ |
| `sim/checks.py` | | checks against the spec, no API calls |
| `sim/plots.py` | | figures for one results folder |
| `sim/theta.py` | | the measures against the share of AI users |
| `sim/report_figures.py`, `sim/consistency.py`, `sim/arms.py` | | figures and tables of the report, from saved results |
| `inputs/m0.csv` | 6 | the starting distribution: 100 shares, built from a survey of researchers; the survey itself is not in this repository |

## Results

The results of the runs behind the report are not in this repository, because the logs with every prompt and response
are large. `sim/report_figures.py` and `sim/consistency.py` expect them in `results/`, under the folder names listed at
the top of `sim/report_figures.py`.

## Notes

- The equilibrium of the non-AI users is computed with [MFGLib](https://github.com/radar-research-lab/MFGLib)
  (online mirror descent) to an exploitability below 10⁻⁴.
- The starting distribution rests on 12 survey responses and the significance scale on machine-learning reviews.
- Runs with Haiku are not exactly repeatable: the model's answers vary between calls.
