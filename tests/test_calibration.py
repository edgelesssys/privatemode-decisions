"""Temperature, calibrate() and prediction sets, on synthetic answers."""

import math
import random

import pytest
from test_inference import make_engine, question

from decisions import SystemOne, calibrate
from decisions.calibration import (
    FAMILY_TEMPERATURES,
    default_temperature,
    fit_temperature,
    peakedness,
    rescale,
    scale,
)
from decisions.types import ChoiceAnswer


def answer(probabilities: dict[str, float]) -> ChoiceAnswer:
    return ChoiceAnswer(choice=max(probabilities, key=probabilities.get),
                        probabilities=probabilities,
                        confidence=peakedness(probabilities.values()))


def sharpened_sample(n: int, temperature: float, seed: int = 0):
    """Answers whose true distribution is known, reported too sharp by
    ``temperature``: the label is drawn from the true distribution."""
    rng = random.Random(seed)
    names = ["a", "b", "c", "d"]
    answers, labels = [], []
    for _ in range(n):
        logits = [rng.gauss(0, 1.5) for _ in names]
        total = sum(math.exp(v) for v in logits)
        true = {k: math.exp(v) / total for k, v in zip(names, logits)}
        labels.append(rng.choices(names, weights=list(true.values()))[0])
        answers.append(answer(scale(true, 1 / temperature)))
    return answers, labels


def test_scale_keeps_the_order_and_sums_to_one():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    for t in (0.5, 1.0, 2.0):
        q = scale(p, t)
        assert math.isclose(sum(q.values()), 1.0)
        assert sorted(q, key=q.get) == sorted(p, key=p.get)
    assert scale(p, 2.0)["a"] < p["a"] < scale(p, 0.5)["a"]


def test_rescale_changes_confidence_not_choice():
    a = answer({"a": 0.9, "b": 0.08, "c": 0.02})
    softer = rescale(a, 2.0)
    assert softer.choice == a.choice
    assert softer.confidence < a.confidence


def test_fit_temperature_recovers_the_sharpening():
    answers, labels = sharpened_sample(4000, temperature=2.0)
    assert fit_temperature(answers, labels) == pytest.approx(2.0, rel=0.1)


def test_calibrate_hits_the_coverage_on_new_data():
    answers, labels = sharpened_sample(1000, temperature=2.0, seed=1)
    calibration = calibrate(answers, labels, coverage=0.9)
    fresh, truth = sharpened_sample(4000, temperature=2.0, seed=2)
    hits = [label in calibration.predict_set(a) for a, label in zip(fresh, truth)]
    assert sum(hits) / len(hits) == pytest.approx(0.9, abs=0.02)
    assert calibration.temperature == pytest.approx(2.0, rel=0.15)


def test_prediction_sets_rank_the_most_likely_first():
    answers, labels = sharpened_sample(500, temperature=1.0, seed=3)
    calibration = calibrate(answers, labels, coverage=0.95)
    a = answer({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})
    chosen = calibration.predict_set(a)
    assert chosen == sorted(chosen, key=lambda k: -a.probabilities[k])
    assert chosen[0] == "a"


def test_calibrate_rejects_labels_that_are_not_options():
    with pytest.raises(ValueError, match="not one of the options"):
        calibrate([answer({"a": 0.6, "b": 0.4})], ["c"])


def test_the_engine_applies_its_temperature():
    raw_engine, _ = make_engine(temperature=1.0)
    soft_engine, _ = make_engine(temperature=3.0)
    raw = raw_engine.system_one("state", {"q": question(5)}).answers["q"]
    soft = soft_engine.system_one("state", {"q": question(5)}).answers["q"]
    assert soft.choice == raw.choice
    assert soft.probabilities[soft.choice] < raw.probabilities[raw.choice]
    assert soft.probabilities == pytest.approx(scale(raw.probabilities, 3.0))


def test_the_default_temperature_follows_the_model_and_options():
    two, many = default_temperature("glm-5.3-flash", 2), default_temperature("glm-5.3-flash", 150)
    assert two > many > 1          # fewer options need more softening
    assert default_temperature("some-other-model", 5) == 1.0
    assert default_temperature("glm-5.3-flash", 5, "sentiment") == FAMILY_TEMPERATURES["glm-5.3-flash"]["sentiment"]
    with pytest.raises(ValueError, match="task family"):
        default_temperature("glm-5.3-flash", 5, "astrology")


def test_a_family_name_sets_the_engine_temperature():
    raw_engine, _ = make_engine(temperature=1.0)
    raw = raw_engine.system_one("state", {"q": question(5)}).answers["q"]
    other = SystemOne(raw_engine.client, "glm-5.3-flash", temperature="sentiment")
    other.oracle = raw_engine.oracle
    soft = other.system_one("state", {"q": question(5)}).answers["q"]
    t = FAMILY_TEMPERATURES["glm-5.3-flash"]["sentiment"]
    assert soft.probabilities == pytest.approx(scale(raw.probabilities, t))
    with pytest.raises(ValueError, match="task family"):
        SystemOne(raw_engine.client, "glm-5.3-flash", temperature="astrology")


def test_a_prediction_set_is_never_empty():
    answers, labels = sharpened_sample(500, temperature=1.0, seed=4)
    calibration = calibrate(answers, labels, coverage=0.5)
    flat = answer({"a": 0.26, "b": 0.25, "c": 0.25, "d": 0.24})
    assert calibration.predict_set(flat)[:1] == ["a"]
