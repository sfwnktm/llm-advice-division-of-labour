# New model: specification so far

Settled: time, approaches, options, significance, depletion, choice of the crowd, the parameter $\theta$, cost of a move, non-AI users, the AI users' rule with $R = 1$, the checks for $R = 1$ (section 15). Not yet: the starting distribution $m_0$, measures.

## 1. Time

$$t \in \{0, 1, \dots, T\}, \qquad T = 20$$

Choices are made at $t = 0, \dots, T-1$. There is no terminal cost.

## 2. Approaches

$$x = (q, g, a) \in E = Q \times D \times A$$

$$|Q| = 5, \qquad |D| = 5, \qquad |A| = 4, \qquad |E| = 100$$

## 3. Distance

Write $x = (x_q, x_g, x_a)$ for the three parts of $x$, and the same for $y$. The distance is the number of parts in which they differ:

$$d(x, y) = \#\{\, p \in \{q, g, a\} : x_p \neq y_p \,\} \in \{0, 1, 2, 3\}$$

## 4. Options

$$\mathcal{A}(x) = \{\, y \in E : d(x, y) \le 1 \,\}$$

$$|\mathcal{A}(x)| = 1 + 4 + 4 + 3 = 12$$

Switching cost of a move from $x$ to $y$:

$$\gamma \, d(x, y)$$

## 5. Significance at the start

$$\kappa_0(x) = 6.7 + \frac{\phi_q + \phi_g + \phi_a}{\sqrt{3}}$$

NK landscape with $K = 2$, each $\phi \sim N(0, 1)$, so

$$\kappa_0(x) \sim N(6.7,\ 1)$$

## 6. Crowd

Share of all researchers on $x$ at $t$:

$$m_t(x) \in [0, 1], \qquad \sum_{x \in E} m_t(x) = 1$$

## 7. Order inside period $t$

1. The crowd $m_t$ is given.
2. Work: every user on $x$ obtains $\kappa_t(x)$.
3. Depletion: $\kappa_{t+1}$ is computed from $\kappa_t$ and $m_t$.
4. Everyone chooses $y \in \mathcal{A}(x)$ at the same time.

## 8. Depletion, no recovery

$$\kappa_{t+1}(x) = \kappa_t(x)\,\bigl(1 - \lambda\, m_t(x)\bigr), \qquad \lambda = 1$$

## 9. Choice: how the crowd moves

$$L_t(x, y) \ge 0, \qquad L_t(x, y) = 0 \ \text{ for } y \notin \mathcal{A}(x)$$

$$\sum_{y \in \mathcal{A}(x)} L_t(x, y) = m_t(x) \quad \text{for every } x \in E$$

$$m_{t+1}(y) = \sum_{x \in E} L_t(x, y)$$

## 10. Populations

Share of AI users, a parameter:

$$\theta \in [0, 1]$$

Non-AI users have share $1 - \theta$. Each population has its own distribution over approaches, $m^N_t$ and $m^A_t$, with $\sum_x m^N_t(x) = \sum_x m^A_t(x) = 1$. The crowd is

$$m_t(x) = (1 - \theta)\, m^N_t(x) + \theta\, m^A_t(x)$$

## 11. Cost of a move

A user on $x$ who moves to $y \in \mathcal{A}(x)$, when next period's crowd is $m_{t+1}$, pays

$$C_t(x, y; m_{t+1}) = \gamma\, d(x, y) - \kappa_{t+1}(y) + c\, m_{t+1}(y)$$

$$\gamma = 0.25, \qquad c = 2$$

Lower cost is better.

## 12. Non-AI users: period equilibrium

Flows of non-AI users:

$$L^N_t(x, y) \ge 0, \qquad L^N_t(x, y) = 0 \ \text{ for } y \notin \mathcal{A}(x)$$

$$\sum_{y \in \mathcal{A}(x)} L^N_t(x, y) = m^N_t(x), \qquad m^N_{t+1}(y) = \sum_{x \in E} L^N_t(x, y)$$

Best response: flows go only to cheapest options,

$$L^N_t(x, y) > 0 \;\Rightarrow\; y \in \arg\min_{y' \in \mathcal{A}(x)} C_t(x, y'; m_{t+1})$$

where the crowd that results is

$$m_{t+1}(y) = (1 - \theta)\, m^N_{t+1}(y) + \theta\, m^A_{t+1}(y)$$

and $m^A_{t+1}$ is given by the AI users' rule (section 13).

## 13. AI users: the rule

The prompt for a group on $x$ at time $t$ lists, for each $y \in \mathcal{A}(x)$ in random order: $d(x,y)$, $\kappa_{t+1}(y)$, $m_t(y)$, and $m_{t-1}(y)$ (not at $t = 0$). It also states the cost rule of section 11. There is no memory and no period number.

Haiku returns one answer per prompt: a choice $y_t(x) \in \mathcal{A}(x)$ and a forecast of next period's crowd. The number of answers per approach and period is $R = 1$. All AI users on $x$ follow that answer:

$$\pi^A_t(y \mid x) = \begin{cases} 1 & \text{if } y = y_t(x) \\ 0 & \text{otherwise} \end{cases}$$

$$L^A_t(x, y) = m^A_t(x)\, \pi^A_t(y \mid x), \qquad m^A_{t+1}(y) = \sum_{x \in E} L^A_t(x, y)$$

The forecast is recorded only. It does not enter the flows.

## 14. Order of one period, both populations

1. The crowd $m_t$ is given.
2. Work: every user on $x$ obtains $\kappa_t(x)$.
3. Depletion: $\kappa_{t+1}$ is computed.
4. AI users: one call per approach with $m^A_t(x) > 0$ gives $L^A_t$ and $m^A_{t+1}$.
5. Non-AI users: the period equilibrium of section 12, given $m^A_{t+1}$, gives $m^N_{t+1}$.
6. The new crowd is $m_{t+1} = (1 - \theta)\, m^N_{t+1} + \theta\, m^A_{t+1}$.

## 15. Checks for $R = 1$

With $R = 1$ every group of AI users on $x$ follows one answer $y_t(x)$. The idea: keep Haiku, and run the same model with other advisers in its place. Only the rule that produces $y_t(x)$ changes. Everything else stays fixed: landscape, $\theta$, non-AI users, option order, and the number of answers per approach ($R = 1$ for all advisers).

**Advisers.** Each one returns a choice $y_t(x) \in \mathcal{A}(x)$.

| Name | Choice $y_t(x)$ |
|---|---|
| Haiku | the answer of Haiku to the prompt of section 13 |
| Stay | $y = x$ |
| Random | $y$ drawn uniformly from $\mathcal{A}(x)$ |
| Naive | the minimiser over $y \in \mathcal{A}(x)$ of $\gamma\, d(x,y) - \kappa_{t+1}(y) + c\, m_t(y)$ |

The Naive adviser is the cost of section 11 with the current crowd $m_t$ in place of the forecast $m_{t+1}$. The non-AI users use the true $m_{t+1}$ (section 12). So Naive and non-AI differ only in this one forecast.

**Design.** Every adviser runs on the same landscapes $\kappa_0$ (10 of them) and the same seeds, for each $\theta$ in the grid. Differences are taken landscape by landscape (paired), then averaged. Each landscape is run several times with different seeds to measure run noise.

**One number for Haiku against Naive.** Over all calls, the match rate is

$$a = \frac{\text{number of calls with } y_t(x) = y^{\text{Naive}}_t(x)}{\text{number of calls}}$$

where $y^{\text{Naive}}_t(x)$ is the Naive choice on the same prompt. In the logs mined so far (thinking on, earlier prompt) $a = 0.988$.

**Possible confounds, and how each is handled.**

| Confound | Handling |
|---|---|
| The landscape differs between arms | same 10 landscapes in every arm, paired differences |
| Run noise | several seeds per landscape, report the spread |
| Option order in the prompt | random order per call, same seeds in all arms, check position share |
| Lumpiness from $R = 1$ | all advisers answer once per $x$, so the lumpiness is in every arm |
| Haiku setting (thinking on or off) | two Haiku arms, nothing else changed |
| Prompt wording | one fixed prompt, version recorded |
| $\theta$ mixes two effects (AI share and the non-AI response) | same grid of $\theta$ for every adviser |
| Invalid answers and retries | count them per arm, report the share |
| Model version and date | record the model ID and date of each run |

