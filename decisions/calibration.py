"""Calibrated probabilities and conformal prediction sets.

Two tiers, measured in ``privatemode-decisions-benchmark``:

* **No labels.** Raw probabilities from one masked token are overconfident.
  Dividing the log probabilities by a temperature ``T > 1`` fixes most of
  that. :class:`~decisions.SystemOne` applies the benchmark's value by
  default, from the number of options (:data:`FORMULAS`) or, if you name
  the kind of task, from its family (:data:`FAMILY_TEMPERATURES`). It is a
  better starting point, not a guarantee: the best ``T`` varies by task.
* **A few hundred labels.** :func:`calibrate` fits a temperature for your
  task and a conformal cutoff; :meth:`Calibration.predict_set` then returns
  the options that contain the right one with the chosen probability, a
  guarantee that holds on average over new inputs drawn like the labelled
  ones. Label a *random* sample: labels collected only from escalated cases
  are skewed toward hard ones and break the guarantee.

Temperature never changes the chosen option, only how sure the answer says
it is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .types import ChoiceAnswer

#: Per model, ``log T = a + b * log(options)``, fitted on the 28 text
#: datasets of the benchmark with every dataset weighted equally. Harder,
#: fewer-option tasks need more softening. Only measured models are listed;
#: others keep their raw probabilities.
FORMULAS: dict[str, tuple[float, float]] = {
    "glm-5.3-flash": (0.962, -0.076),
    "glm-flash-latest": (0.962, -0.076),
}

#: Per model and task family, one temperature fitted on that family's
#: datasets. Closer to a task's own best value than the formula, if you can
#: say what kind of task it is.
FAMILY_TEMPERATURES: dict[str, dict[str, float]] = {
    "glm-5.3-flash": {"intent": 1.76, "legal": 2.24, "moderation": 2.96, "nli": 1.87,
                      "qa": 1.76, "sentiment": 2.93, "topic": 2.60},
}
FAMILY_TEMPERATURES["glm-flash-latest"] = FAMILY_TEMPERATURES["glm-5.3-flash"]

_FLOOR = 1e-300


def default_temperature(model: str, options: int, family: str | None = None) -> float:
    """The benchmark's temperature for ``model``: by task family if one is
    given, else from the number of options; 1 (raw) for unmeasured models."""
    if family is not None:
        by_family = FAMILY_TEMPERATURES.get(model, {})
        if family not in by_family:
            known = ", ".join(sorted(by_family)) or "none for this model"
            raise ValueError(f"no temperature for task family {family!r} (known: {known})")
        return by_family[family]
    if model not in FORMULAS:
        return 1.0
    a, b = FORMULAS[model]
    return math.exp(a + b * math.log(max(options, 2)))


def scale(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    """``p ** (1 / T)``, renormalized."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1:
        return dict(probabilities)
    logs = {k: math.log(max(p, _FLOOR)) / temperature for k, p in probabilities.items()}
    top = max(logs.values())
    weights = {k: math.exp(v - top) for k, v in logs.items()}
    total = math.fsum(weights.values())
    return {k: w / total for k, w in weights.items()}


def peakedness(probabilities: Iterable[float]) -> float:
    """1 for all mass on one option, 0 for a uniform spread."""
    values = list(probabilities)
    if len(values) < 2:
        return 1.0
    entropy = math.fsum(-p * math.log(p) for p in values if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(values))))


def rescale(answer: ChoiceAnswer, temperature: float) -> ChoiceAnswer:
    """The same answer with its probabilities and confidence rescaled."""
    probabilities = scale(answer.probabilities, temperature)
    return replace(answer, probabilities=probabilities,
                   confidence=peakedness(probabilities.values()))


def _nll(logs: Sequence[tuple[list[float], int]], temperature: float) -> float:
    total = 0.0
    for row, gold in logs:
        scaled = [v / temperature for v in row]
        top = max(scaled)
        total += top + math.log(math.fsum(math.exp(v - top) for v in scaled)) - scaled[gold]
    return total / len(logs)


def fit_temperature(answers: Sequence[ChoiceAnswer], labels: Sequence[str]) -> float:
    """The temperature minimizing negative log-likelihood on labelled answers.

    NLL is convex in ``1 / T``, so a golden-section search on ``log T`` over
    [0.05, 50] finds it.
    """
    logs = []
    for answer, label in zip(answers, labels, strict=True):
        names = list(answer.probabilities)
        if label not in answer.probabilities:
            raise ValueError(f"label {label!r} is not one of the options {names}")
        logs.append(([math.log(max(answer.probabilities[n], _FLOOR)) for n in names],
                     names.index(label)))
    if not logs:
        raise ValueError("no labelled answers to fit on")
    a, b = math.log(0.05), math.log(50.0)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll(logs, math.exp(c)), _nll(logs, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _nll(logs, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _nll(logs, math.exp(d))
    return math.exp((a + b) / 2)


@dataclass(frozen=True)
class Calibration:
    """A per-task temperature and conformal cutoff, from :func:`calibrate`.

    ``cutoff`` applies to ``1 - p`` after the temperature: an option is in
    the prediction set when its probability is at least ``1 - cutoff``.
    """

    temperature: float
    cutoff: float
    coverage: float
    examples: int

    def apply(self, answer: ChoiceAnswer) -> ChoiceAnswer:
        """The answer with this task's temperature applied."""
        return rescale(answer, self.temperature)

    def predict_set(self, answer: ChoiceAnswer) -> list[str]:
        """The options that contain the right one with probability
        ``coverage``, most likely first. One option means it can be
        automated at that level; several mean a person should choose.

        Never empty: where no option clears the cutoff, the most likely one
        is returned, which only makes the coverage higher."""
        probabilities = scale(answer.probabilities, self.temperature)
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)
        return [name for name in ranked if 1 - probabilities[name] <= self.cutoff] or ranked[:1]


def calibrate(answers: Sequence[ChoiceAnswer], labels: Sequence[str], *,
              coverage: float = 0.9) -> Calibration:
    """Fit a temperature and a conformal cutoff on labelled answers.

    ``answers`` are what :meth:`~decisions.SystemOne.system_one` returned for
    one question on a random sample of inputs, ``labels`` the right option
    for each. A few hundred make the coverage reliable: on the benchmark, a
    90% target landed between 85% and 95% with 100 labels and between 87%
    and 92% with 500 (5th to 95th percentile). The guarantee needs new inputs
    to come from the same distribution as the labelled ones.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    temperature = fit_temperature(answers, labels)
    scores = sorted(1 - scale(a.probabilities, temperature)[label]
                    for a, label in zip(answers, labels, strict=True))
    rank = math.ceil((len(scores) + 1) * coverage)
    cutoff = 1.0 if rank > len(scores) else scores[rank - 1]
    return Calibration(temperature=temperature, cutoff=cutoff, coverage=coverage,
                       examples=len(scores))
