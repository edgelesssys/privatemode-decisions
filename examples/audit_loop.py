"""An audit loop: check a task's calibration on fresh labels, refit on drift.

    python examples/audit_loop.py decisions.jsonl labels.jsonl calibration.json

``calibrate()`` gives guarantees for inputs drawn like the ones it was fitted
on. Traffic changes, and the best temperature follows difficulty, so the
guarantees have to be checked where they are used. The loop:

1. Log every decision with its **raw answer**, the probabilities before any
   task calibration (the library's default temperature is fine, as long as
   it is the same throughout).
2. Now and then, draw a **random** sample of logged decisions and have a
   person label them. Not the escalated ones only: those are the hard cases,
   and a calibration fitted or checked on them promises nothing about the
   rest.
3. ``evaluate()`` the current calibration on that sample. If coverage or the
   error among automated answers is off by more than sampling explains,
   refit with ``calibrate()`` on the sample and keep it for the next round.

The files are JSON lines. ``decisions.jsonl``: ``{"id": ..., "probabilities":
{...}}`` per decision, as logged. ``labels.jsonl``: ``{"id": ..., "label":
...}`` for the audited ones. ``calibration.json`` holds the current
calibration and is rewritten after a refit.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

from decisions import Calibration, ChoiceAnswer, calibrate, evaluate

COVERAGE = 0.9
MAX_ERROR = 0.05
#: Labels per audit. A few hundred make coverage and the error bound
#: reliable; with 100, a 90% target lands between 85% and 95%.
SAMPLE = 300


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answer(probabilities: dict[str, float]) -> ChoiceAnswer:
    return ChoiceAnswer(choice=max(probabilities, key=probabilities.get),
                        probabilities=probabilities, confidence=0.0)


def draw(decisions: list[dict], seed: int | None = None) -> list[dict]:
    """The decisions to send to a person: a uniform random sample."""
    return random.Random(seed).sample(decisions, min(SAMPLE, len(decisions)))


def drifted(report: dict[str, float], n: int) -> list[str]:
    """What is off by more than a binomial sample of ``n`` explains (about
    two standard errors)."""
    problems = []
    slack = 2 * math.sqrt(COVERAGE * (1 - COVERAGE) / n)
    if report["coverage"] < COVERAGE - slack:
        problems.append(f"coverage {report['coverage']:.1%} < {COVERAGE:.0%}")
    automated = round(report["automated"] * n)
    if automated:
        slack = 2 * math.sqrt(MAX_ERROR * (1 - MAX_ERROR) / automated)
        if report["automated_error"] > MAX_ERROR + slack:
            problems.append(f"automated error {report['automated_error']:.1%} > {MAX_ERROR:.0%}")
    return problems


def main() -> None:
    decisions_path, labels_path, calibration_path = map(Path, sys.argv[1:4])
    decisions = {d["id"]: d for d in load(decisions_path)}
    labelled = [(decisions[row["id"]], row["label"]) for row in load(labels_path)
                if row["id"] in decisions]
    answers = [answer(d["probabilities"]) for d, _ in labelled]
    labels = [label for _, label in labelled]

    if calibration_path.exists():
        current = Calibration(**json.loads(calibration_path.read_text()))
        report = evaluate(answers, labels, calibration=current)
        print(json.dumps({k: round(v, 4) for k, v in report.items()}))
        problems = drifted(report, len(labels))
        if not problems:
            print("calibration holds; keep it")
            return
        print("drift:", "; ".join(problems))
    # Refit on this audit's labels. They are a fresh random sample, so the
    # new calibration's guarantees hold for traffic like today's.
    fitted = calibrate(answers, labels, coverage=COVERAGE, max_error=MAX_ERROR)
    calibration_path.write_text(json.dumps(asdict(fitted), indent=1))
    print(f"refitted on {len(labels)} labels: T = {fitted.temperature:.2f}, "
          f"threshold {fitted.threshold:.3f}")


if __name__ == "__main__":
    main()
