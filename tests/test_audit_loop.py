"""examples/audit_loop.py: a refit across a change of temperature."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_calibration import biased_sample

from decisions import Calibration
from decisions.calibration import rescale

spec = importlib.util.spec_from_file_location(
    "audit_loop", Path(__file__).resolve().parent.parent / "examples" / "audit_loop.py")
audit_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_loop)


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def logged(answers, temperatures) -> list[dict]:
    return [{"id": i, "probabilities": rescale(a, t).probabilities, "temperature": t}
            for i, (a, t) in enumerate(zip(answers, temperatures))]


def test_the_refit_takes_a_sample_across_a_change_of_temperature(tmp_path, capsys):
    """The default moved from 2 to 3 halfway through the logged decisions:
    the stored calibration no longer fits, and the refit brings the sample to
    the newest temperature instead of stopping at the mix."""
    answers, labels = biased_sample(300, seed=21)
    decisions = write(tmp_path / "decisions.jsonl", logged(answers, [2.0] * 150 + [3.0] * 150))
    labelled = write(tmp_path / "labels.jsonl", [{"id": i, "label": y} for i, y in enumerate(labels)])
    stored = tmp_path / "calibration.json"
    old = audit_loop.calibrate([rescale(a, 2.0) for a in answers], labels)
    audit_loop.save(old, stored)
    audit_loop.check(decisions, labelled, stored)
    assert "refitted" in capsys.readouterr().out
    fitted = audit_loop.restore(stored)
    assert fitted.base_temperature == 3.0
    assert fitted.examples == 300


def test_rescaling_to_the_newest_temperature_matches_asking_at_it():
    answers, _ = biased_sample(5, seed=22)
    at_two = [rescale(a, 2.0) for a in answers]
    moved = audit_loop.at_one_temperature(at_two, 3.0)
    for a, b in zip(moved, (rescale(a, 3.0) for a in answers)):
        assert a.temperature == 3.0
        assert a.probabilities == pytest.approx(b.probabilities)


def test_a_stored_calibration_is_standard_json(tmp_path):
    answers, labels = biased_sample(100, seed=23)
    fitted = audit_loop.calibrate(answers, labels)          # no max_error: threshold inf
    audit_loop.save(fitted, tmp_path / "c.json")
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["threshold"] is None
    assert Calibration.from_dict(data) == fitted
