"""Temperature, calibrate() and prediction sets, on synthetic answers."""

import math
import random

import pytest
from test_inference import make_engine, question

from decisions import SystemOne, calibrate, evaluate
from decisions.calibration import (
    FAMILY_TEMPERATURES,
    default_temperature,
    fit_temperature,
    fit_temperature_bias,
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


def test_automation_keeps_its_error_promise():
    answers, labels = sharpened_sample(2000, temperature=1.0, seed=5)
    calibration = calibrate(answers, labels, max_error=0.10)
    assert calibration.threshold < math.inf
    fresh, truth = sharpened_sample(8000, temperature=1.0, seed=6)
    automated = [(a, t) for a, t in zip(fresh, truth) if calibration.automate(a)]
    assert automated
    errors = sum(a.choice != t for a, t in automated) / len(automated)
    assert errors <= 0.10
    assert not calibrate(answers, labels).automate(fresh[0])   # no max_error, no automation


def test_per_class_cutoffs_cover_a_rare_class():
    rng = random.Random(7)

    def sample(n):
        answers, labels = [], []
        for _ in range(n):
            rare = rng.random() < 0.1
            # The rare class looks like the common one most of the time.
            p = rng.uniform(0.3, 0.6) if rare else rng.uniform(0.0, 0.3)
            answers.append(answer({"rare": p, "common": 1 - p}))
            labels.append("rare" if rare else "common")
        return answers, labels

    answers, labels = sample(3000)
    fresh, truth = sample(6000)
    for per_class, low, high in ((False, 0.0, 0.85), (True, 0.87, 1.0)):
        # A bias would shift probability towards the rare class by itself;
        # this is about the cutoffs.
        calibration = calibrate(answers, labels, coverage=0.9, per_class=per_class, bias=False)
        hits = [t in calibration.predict_set(a) for a, t in zip(fresh, truth) if t == "rare"]
        assert low <= sum(hits) / len(hits) <= high


def test_few_labels_are_pulled_towards_no_change():
    answers, labels = sharpened_sample(15, temperature=2.0, seed=8)
    free = fit_temperature(answers, labels, shrinkage=0)
    pulled = fit_temperature(answers, labels)
    assert abs(math.log(pulled)) < abs(math.log(free))
    many, many_labels = sharpened_sample(4000, temperature=2.0, seed=9)
    assert fit_temperature(many, many_labels) == pytest.approx(
        fit_temperature(many, many_labels, shrinkage=0), rel=0.01)


def test_evaluate_reports_calibration_sets_and_automation():
    answers, labels = sharpened_sample(3000, temperature=2.0, seed=10)
    raw = evaluate(answers, labels)
    assert raw["overconfidence"] > 0.05 and raw["ece"] > 0.05
    calibration = calibrate(answers[:1500], labels[:1500], coverage=0.9, max_error=0.1)
    fixed = evaluate(answers[1500:], labels[1500:], calibration=calibration)
    assert fixed["ece"] < raw["ece"] / 2
    assert fixed["coverage"] == pytest.approx(0.9, abs=0.03)
    assert fixed["automated_error"] <= 0.1
    assert 0 < fixed["automated"] < 1
    assert set(raw) == {"accuracy", "confidence", "overconfidence", "ece"}


def test_aliases_share_their_model_temperature():
    assert default_temperature("glm-flash-latest", 7) == default_temperature("glm-5.3-flash", 7) > 1


def test_options_with_too_few_labels_are_always_included():
    answers, labels = sharpened_sample(60, temperature=1.0, seed=11)
    counts = {name: labels.count(name) for name in "abcd"}
    calibration = calibrate(answers, labels, coverage=0.9, per_class=True)
    for name, count in counts.items():
        assert (name in calibration.always_included) == (count < 9)
    rare = [name for name, count in counts.items() if count < 9]
    for answer_ in answers[:20]:
        assert set(rare) <= set(calibration.predict_set(answer_))
    assert calibrate(answers, labels).always_included == ()


def biased_sample(n: int, seed: int = 0):
    """Calibrated answers, except that the model moves probability from
    ``b`` to ``a``: log p(a) is raised by 1.5 before renormalizing."""
    rng = random.Random(seed)
    names = ["a", "b", "c"]
    answers, labels = [], []
    for _ in range(n):
        logits = [rng.gauss(0, 1.5) for _ in names]
        total = sum(math.exp(v) for v in logits)
        true = {k: math.exp(v) / total for k, v in zip(names, logits)}
        labels.append(rng.choices(names, weights=list(true.values()))[0])
        skewed = {k: v * (math.exp(1.5) if k == "a" else 1.0) for k, v in true.items()}
        total = sum(skewed.values())
        answers.append(answer({k: v / total for k, v in skewed.items()}))
    return answers, labels


def test_bias_recovers_a_preferred_option():
    answers, labels = biased_sample(4000)
    t, bias = fit_temperature_bias(answers, labels)
    assert t == pytest.approx(1.0, abs=0.1)
    assert bias["a"] - bias["b"] == pytest.approx(-1.5, abs=0.2)
    assert sum(bias.values()) == pytest.approx(0.0, abs=1e-9)


def test_bias_is_pulled_towards_zero_with_few_labels():
    answers, labels = biased_sample(10, seed=3)
    _, weak = fit_temperature_bias(answers, labels, strength=100.0)
    _, free = fit_temperature_bias(answers, labels, strength=0.01)
    assert max(map(abs, weak.values())) < max(map(abs, free.values()))


def test_calibrate_with_bias_changes_answers_and_gains_accuracy():
    answers, labels = biased_sample(1000, seed=1)
    fresh, truth = biased_sample(4000, seed=2)
    with_bias = calibrate(answers, labels)
    alone = calibrate(answers, labels, bias=False)
    assert with_bias.bias and not alone.bias
    corrected = [with_bias.apply(a) for a in fresh]
    assert any(c.choice != a.choice for c, a in zip(corrected, fresh))
    accuracy = sum(c.choice == t for c, t in zip(corrected, truth)) / len(truth)
    raw = sum(a.choice == t for a, t in zip(fresh, truth)) / len(truth)
    assert accuracy > raw + 0.02
    assert evaluate(fresh, truth, calibration=with_bias)["accuracy"] == pytest.approx(accuracy)
    assert evaluate(fresh, truth, calibration=alone)["accuracy"] == pytest.approx(raw)


def test_calibrate_with_bias_keeps_coverage_and_the_error_bound():
    answers, labels = biased_sample(1000, seed=4)
    fresh, truth = biased_sample(6000, seed=5)
    calibration = calibrate(answers, labels, coverage=0.9, max_error=0.10)
    report = evaluate(fresh, truth, calibration=calibration)
    assert report["coverage"] == pytest.approx(0.9, abs=0.03)
    assert 0 < report["automated"] and report["automated_error"] <= 0.10


def test_measured_models_and_their_aliases_have_defaults():
    for alias, model in (("kimi-latest", "kimi-k2.6"), ("glm-latest", "glm-5.3"),
                         ("glm-flash-latest", "glm-5.3-flash")):
        assert default_temperature(alias, 4) == default_temperature(model, 4) > 1
    assert default_temperature("some-unmeasured-model", 4) == 1.0


def test_out_of_fold_calibration_does_not_depend_on_the_label_order():
    answers, labels = biased_sample(500, seed=11)
    shuffled = calibrate(answers, labels, max_error=0.15)
    ranked = sorted(zip(answers, labels), key=lambda al: al[1])     # grouped by class
    grouped = calibrate([a for a, _ in ranked], [l for _, l in ranked], max_error=0.15)
    assert grouped.threshold < math.inf
    assert grouped.threshold == pytest.approx(shuffled.threshold, abs=0.05)
    assert grouped.cutoffs["*"] == pytest.approx(shuffled.cutoffs["*"], abs=0.05)


def test_a_calibration_refuses_answers_unlike_its_own():
    answers, labels = biased_sample(300, seed=12)
    at_default = [rescale(a, 2.0) for a in answers]           # as SystemOne reports them
    calibration = calibrate(at_default, labels)
    assert calibration.base_temperature == pytest.approx(2.0)
    assert calibration.options == ("a", "b", "c")
    calibration.apply(rescale(answers[0], 2.0))                # the same kind: fine
    with pytest.raises(ValueError, match="temperature"):
        calibration.apply(rescale(answers[0], 2.5))            # the default moved
    renamed = answer({"a": 0.5, "b": 0.3, "d": 0.2})
    with pytest.raises(ValueError, match="options"):
        calibration.predict_set(renamed)
    # Answers without a known temperature (e.g. from a log) are only checked for options.
    calibration.automate(answers[0])


def test_calibrate_refuses_answers_at_different_temperatures():
    answers, labels = biased_sample(100, seed=13)
    mixed = [rescale(a, 2.0 if i % 2 else 3.0) for i, a in enumerate(answers)]
    with pytest.raises(ValueError, match="different temperatures"):
        calibrate(mixed, labels)


@pytest.mark.parametrize("max_error", [None, 0.2])
def test_a_calibration_survives_a_json_round_trip(max_error):
    """Through standard JSON (no Infinity), with and without a threshold."""
    import json

    from decisions import Calibration

    answers, labels = biased_sample(300, seed=14)
    fitted = calibrate([rescale(a, 2.0) for a in answers], labels, per_class=True,
                       max_error=max_error)
    restored = Calibration.from_dict(json.loads(json.dumps(fitted.to_dict(), allow_nan=False)))
    assert restored == fitted
    assert restored.predict_set(rescale(answers[0], 2.0)) == fitted.predict_set(rescale(answers[0], 2.0))


def test_too_few_labels_for_a_bias_fit_the_temperature_alone():
    from decisions.calibration import MIN_BIAS_LABELS

    answers, labels = biased_sample(MIN_BIAS_LABELS - 1, seed=15)
    assert calibrate(answers, labels).bias == {}
    answers, labels = biased_sample(MIN_BIAS_LABELS, seed=15)
    assert calibrate(answers, labels).bias


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_temperature_must_be_a_positive_finite_number(bad):
    with pytest.raises(ValueError):
        scale({"a": 0.6, "b": 0.4}, bad)
    with pytest.raises(ValueError):
        make_engine(temperature=bad)


def test_extreme_logits_stay_within_the_searched_temperatures():
    """Near one-hot answers with many exact zeros once drove the joint fit's
    line search to exp overflow."""
    from decisions.types import ChoiceAnswer

    rng = random.Random(24)
    names = [f"o{i}" for i in range(20)]
    answers, labels = [], []
    for _ in range(60):
        logits = [rng.gauss(0, 800) for _ in names]
        top = max(logits)
        weights = [math.exp(v - top) for v in logits]
        total = sum(weights)
        probabilities = {n: w / total for n, w in zip(names, weights)}
        answers.append(ChoiceAnswer(choice=max(probabilities, key=probabilities.get),
                                    probabilities=probabilities, confidence=1.0))
        labels.append(rng.choice(names))
    fitted = calibrate(answers, labels)
    assert 0.05 <= fitted.temperature <= 50


@pytest.mark.parametrize("delta", [0.0, 1.0, 1.5, -0.1])
def test_delta_must_be_a_probability(delta):
    answers, labels = biased_sample(50, seed=25)
    with pytest.raises(ValueError, match="delta"):
        calibrate(answers, labels, max_error=0.1, delta=delta)


def test_the_automation_threshold_certifies_levels_in_order():
    from decisions.calibration import _automation_threshold

    confidences = [1 - i / 1000 for i in range(100)]
    # All right: every level passes, down to the least confident answer.
    assert _automation_threshold(confidences, [True] * 100, 0.1, 0.1) == confidences[-1]
    # The first level that can pass (22 answers at 10%, here the top 25)
    # holds errors: nothing is certified, even if the rest is clean.
    right = [i >= 10 for i in range(100)]
    assert _automation_threshold(confidences, right, 0.1, 0.1) == math.inf
    # Errors from the 61st answer on: the top 60 pass, the top 65 (5 errors)
    # fail, so the sequence stops there and keeps the top 60.
    right = [i < 60 for i in range(100)]
    assert _automation_threshold(confidences, right, 0.1, 0.1) == confidences[59]
    # Too few answers for any level: nothing.
    assert _automation_threshold(confidences[:20], [True] * 20, 0.1, 0.1) == math.inf


def test_the_automation_threshold_never_splits_equal_confidences():
    from decisions.calibration import _automation_threshold

    # 40 answers share the top confidence; a level can't stop inside them.
    confidences = [0.99] * 40 + [0.5 - i / 1000 for i in range(60)]
    right = [True] * 100
    chosen = _automation_threshold(confidences, right, 0.1, 0.1)
    assert chosen == confidences[-1]
    right = [True] * 40 + [False] * 60
    assert _automation_threshold(confidences, right, 0.1, 0.1) == 0.99


@pytest.mark.parametrize("n, expected", [(8, 1.0), (9, 0.9)])
def test_the_conformal_cutoff_needs_nine_scores_at_ninety_percent(n, expected):
    from decisions.calibration import _cutoff

    scores = [i / 10 for i in range(1, n + 1)]      # 0.1 ... 0.n
    assert _cutoff(scores, 0.9) == pytest.approx(expected)
