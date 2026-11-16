"""A language model as adviser (spec section 13): the prompt, one call through the ant CLI, and the answer.

Advisers: "haiku" and "haiku_thinking": Claude Haiku 5.5 with thinking off or on (adaptive). Same prompt and settings.

The ant CLI holds the API credentials; this code never reads or writes them.
"""
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .parameters import C, EFFORT, GAMMA, MAX_ATTEMPTS, MAX_PARALLEL, MAX_TOKENS, MAX_TOKENS_THINKING, MODELS, THINKING_OFF
from .world import ANALYSIS, APPROACHES, DATA_GATHERING, DISTANCE, QUESTION_TYPES

ANT = shutil.which("ant") or str(Path.home() / "go" / "bin" / "ant")  # Anthropic's command-line client
KEYS = [f"option_{i:02d}" for i in range(1, 13)]

SCHEMA = {
    "type": "object",
    "properties": {
        "forecast": {
            "type": "object",
            "properties": {key: {"type": "number"} for key in KEYS},
            "required": KEYS,
            "additionalProperties": False,
        },
        "option": {"type": "integer", "enum": list(range(1, 13))},
    },
    "required": ["forecast", "option"],
    "additionalProperties": False,
}


def describe(x):
    q, g, a = APPROACHES[x]
    return f"question type: {QUESTION_TYPES[q]}; data gathering: {DATA_GATHERING[g]}; analysis: {ANALYSIS[a]}"


def build_prompt(call, shown):
    """The prompt for a group on x at time t, with the options in the order `shown`.

    Based on the prompt of the earlier R = 1 runs (sim2, "informed"), with the two additions of spec section 13:
    the share last period (not at t = 0) and the forecast. No period number. With memory (an extension of the
    spec), the group's previous studies are listed after its current approach.
    """
    first = call.m_prev is None
    q, g, a = APPROACHES[call.x]
    given = ("the number of parts changed, the significance left on that approach (work on an approach uses up its "
             "significance) and the share of all researchers working on it now")
    if not first:
        given = given.replace(" and the share of all researchers working on it now",
                              ", the share of all researchers working on it now and the share of all researchers "
                              "who worked on it last period")
    lines = []
    for i, y in enumerate(shown, start=1):
        line = (f"{i}. {describe(y)}; parts changed: {DISTANCE[call.x, y]}; "
                f"significance left: {call.kappa_next[y]:.3f}; share of all researchers now: {call.m_now[y]:.4f}")
        if not first:
            line += f"; share of all researchers last period: {call.m_prev[y]:.4f}"
        lines.append(line)
    options = "\n".join(lines)
    remembered = ""
    if call.history and call.show_history:
        studies = [f"- {describe(x)}; significance found: {kappa:.3f}; share of all researchers there: {share:.4f}"
                   for x, kappa, share in call.history]
        remembered = "Before that, its previous studies were, oldest first:\n" + "\n".join(studies) + "\n"

    return f"""Many research groups in a field choose research approaches. An approach has three parts: the type of \
research question, the data-gathering method and the analysis method.
A research group has just completed a study with the following approach.
- Type of research question: {QUESTION_TYPES[q]}
- Data-gathering method: {DATA_GATHERING[g]}
- Analysis method: {ANALYSIS[a]}
{remembered}
In the next period the group can keep its approach or change one of its three parts. For each option you are \
given {given}. The value of the group's next study is: significance left - {C} x share of all researchers on that \
approach next period - {GAMMA} x parts changed. Other groups choose at the same time, so next period's shares can \
differ from the current ones.

Which approach should the group's next study take? Choose exactly one of these options:
{options}

First forecast, for each option, the share of all researchers who will work on that approach next period (a number \
between 0 and 1). Then choose exactly one option.
Answer only with JSON: {{"forecast": {{"option_01": <share>, ..., "option_12": <share>}}, "option": <number>}}"""


def send(prompt, model, thinking, tries=10):
    """One call to the model, with thinking off or on (adaptive). Retries on errors such as rate limits.
    Returns the full response."""
    args = [ANT, "messages", "create", "--model", model,
            "--max-tokens", str(MAX_TOKENS_THINKING if thinking else MAX_TOKENS),
            "--thinking", json.dumps({"type": "adaptive" if thinking else THINKING_OFF[model]}),
            "--output-config", json.dumps({"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}}),
            "--message", json.dumps({"role": "user", "content": prompt})]
    for attempt in range(tries):
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return json.loads(result.stdout)
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"{model} call failed {tries} times: {(result.stdout + result.stderr)[:500]}")


def parse(response):
    """Return (position 1-12, forecast by position) or None if the answer is not valid."""
    text = next((block["text"] for block in response.get("content", []) if block["type"] == "text"), "")
    try:
        answer = json.loads(text)
        position = int(answer["option"])
        forecast = [float(answer["forecast"][key]) for key in KEYS]
    except (ValueError, KeyError, TypeError):
        return None
    return (position, forecast) if 1 <= position <= 12 else None


def advise(call, model, thinking):
    """Ask until the answer is valid, each time with a new option order. Returns the choice and the record."""
    attempts = []
    for attempt in range(MAX_ATTEMPTS):
        shown = call.shown(attempt)
        prompt = build_prompt(call, shown)
        response = send(prompt, model, thinking)
        parsed = parse(response)
        attempts.append({"shown": shown.tolist(), "prompt": prompt, "response": response, "valid": parsed is not None})
        if parsed is not None:
            position, forecast = parsed
            return {"y": int(shown[position - 1]), "valid": True, "position": position,
                    "forecast": dict(zip(shown.tolist(), forecast)), "attempts": attempts}
    return {"y": call.x, "valid": False, "position": None, "forecast": None, "attempts": attempts}


def ask_all(calls, model, thinking):
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        answers = pool.map(lambda call: advise(call, model, thinking), calls)
        return list(tqdm(answers, total=len(calls), desc=f"  {model} calls", unit="call", leave=False))


def model_of(adviser):
    """'haiku_thinking' -> ('claude-haiku-5-5', True)."""
    return MODELS[adviser.split("_")[0]], adviser.endswith("_thinking")


def language_model_adviser(adviser):
    model, thinking = model_of(adviser)
    return lambda calls: ask_all(calls, model, thinking)
