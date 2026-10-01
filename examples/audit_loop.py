"""An audit loop: check a task's calibration on fresh labels, refit on drift.

    python examples/audit_loop.py draw decisions.jsonl to_label.jsonl [-n 300]
    python examples/audit_loop.py check decisions.jsonl labels.jsonl calibration.json

``calibrate()`` gives guarantees for inputs drawn like the ones it was fitted
on. Traffic changes, and the best temperature follows difficulty, so the
guarantees have to be checked where they are used. The loop:

1. Log every decision with its answer as ``system_one`` returned it: the
   probabilities, the temperature they were reported at and the model that
   answered (``answer.model``), before any task calibration. A stored
   calibration refuses answers from another model (an alias that moved) or
   at another temperature (regenerated defaults), so refit then.
2. ``draw``: pick a **random** sample of logged decisions for a person to
   label. Not the escalated ones only: those are the hard cases, and a
   calibration fitted or checked on them promises nothing about the rest.
3. ``check``: ``evaluate()`` the current calibration on the labelled sample.
   If coverage or the error among automated answers is off by more than
   sampling explains, refit with ``calibrate()`` on the sample and keep it
   for the next round. Without a calibration yet, the first run fits one.

The files are JSON lines. ``decisions.jsonl``: ``{"id": ..., "probabilities":
{...}, "temperature": ..., "model": ...}`` per decision, as logged, oldest
first. ``to_label.jsonl`` gets the drawn decisions; ``labels.jsonl`` holds
``{"id": ..., "label": ...}`` for the labelled ones. ``calibration.json``
holds the current calibration and is rewritten after a refit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path

from decisions import Calibration, ChoiceAnswer, calibrate, evaluate
from decisions.calibration import rescale

#: For a first fit; afterwards the stored calibration's own are kept.
COVERAGE = 0.9
MAX_ERROR = 0.05
#: Labels per audit. A few hundred make coverage and the error bound
#: reliable; with 100, a 90% target lands between 85% and 95%.
SAMPLE = 300


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answer(decision: dict) -> ChoiceAnswer:
    probabilities = decision["probabilities"]
    return ChoiceAnswer(choice=max(probabilities, key=probabilities.get),
                        probabilities=probabilities, confidence=0.0,
                        temperature=decision.get("temperature"), model=decision.get("model"))


def save(calibration: Calibration, path: Path) -> None:
    path.write_text(json.dumps(calibration.to_dict(), indent=1))


def restore(path: Path) -> Calibration:
    return Calibration.from_dict(json.loads(path.read_text()))


def at_one_temperature(answers: list[ChoiceAnswer], target: float | None) -> list[ChoiceAnswer]:
    """The answers at the temperature of the newest decision. A sample that
    spans a regenerated default mixes temperatures of one model, which
    ``calibrate()`` refuses; temperatures compose, so each answer is
    rescaled by ``target / its own``. (Answers from another model are
    dropped before this: rescaling can't turn them into this model's.)"""
    if target is None:
        if len({a.temperature for a in answers}) > 1:
            raise SystemExit("the labelled answers are at different temperatures and the "
                             "newest decision has none logged; log the temperature")
        return answers
    return [a if a.temperature in (None, target)
            else replace(rescale(a, target / a.temperature), temperature=target)
            for a in answers]


def draw(decisions: list[dict], n: int, seed: int | None = None) -> list[dict]:
    """The decisions to send to a person: a uniform random sample."""
    return random.Random(seed).sample(decisions, min(n, len(decisions)))


def drifted(report: dict[str, float], calibration: Calibration, n: int) -> list[str]:
    """What is off by more than a binomial sample of ``n`` explains (about
    two standard errors)."""
    problems = []
    target = calibration.coverage
    if report["coverage"] < target - 2 * math.sqrt(target * (1 - target) / n):
        problems.append(f"coverage {report['coverage']:.1%} < {target:.0%}")
    automated = round(report["automated"] * n)
    bound = calibration.max_error
    if bound is not None and automated:
        if report["automated_error"] > bound + 2 * math.sqrt(bound * (1 - bound) / automated):
            problems.append(f"automated error {report['automated_error']:.1%} > {bound:.0%}")
    return problems


def check(decisions_path: Path, labels_path: Path, calibration_path: Path) -> None:
    logged = load(decisions_path)
    decisions = {d["id"]: d for d in logged}
    labelled = [(decisions[row["id"]], row["label"]) for row in load(labels_path)
                if row["id"] in decisions]
    # Refit only on the model that answers now: after an alias moved, the
    # older model's answers say nothing about the new one. Decisions logged
    # without a model (before it was logged) could be from any model, so
    # they go too once the newest decision names one.
    model = logged[-1].get("model") if logged else None
    if model is not None:
        kept = [(d, label) for d, label in labelled if d.get("model") == model]
        if len(kept) < len(labelled):
            print(f"{len(labelled) - len(kept)} labelled decisions are from another model "
                  f"than {model}, or don't say; left out")
        labelled = kept
    if not labelled:
        raise SystemExit("no labelled decisions from the current model")
    answers = [answer(d) for d, _ in labelled]
    newest = logged[-1].get("temperature") if logged else None
    labels = [label for _, label in labelled]
    unknown = sorted({label for label in labels} - set(answers[0].probabilities))
    if unknown:
        raise SystemExit(f"labels {unknown} are not options of these decisions; fix the labels")
    coverage, max_error = COVERAGE, MAX_ERROR
    if calibration_path.exists():
        current = restore(calibration_path)
        coverage, max_error = current.coverage, current.max_error
        try:
            report = evaluate(answers, labels, calibration=current)
        except ValueError as mismatch:      # other options, or another temperature
            print(f"calibration doesn't fit these answers ({mismatch}); refitting")
        else:
            print(json.dumps({k: round(v, 4) for k, v in report.items()}))
            problems = drifted(report, current, len(labels))
            if not problems:
                print("calibration holds; keep it")
                return
            print("drift:", "; ".join(problems))
    # Refit on this audit's labels. They are a fresh random sample, so the
    # new calibration's guarantees hold for traffic like today's.
    fitted = calibrate(at_one_temperature(answers, newest), labels,
                       coverage=coverage, max_error=max_error)
    save(fitted, calibration_path)
    print(f"refitted on {len(labels)} labels: T = {fitted.temperature:.2f}, "
          f"threshold {fitted.threshold:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("draw", help="a random sample of decisions to label")
    d.add_argument("decisions", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("-n", type=int, default=SAMPLE)
    c = commands.add_parser("check", help="evaluate the calibration, refit on drift")
    c.add_argument("decisions", type=Path)
    c.add_argument("labels", type=Path)
    c.add_argument("calibration", type=Path)
    args = parser.parse_args()
    if args.command == "draw":
        rows = draw(load(args.decisions), args.n)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {len(rows)} decisions to label to {args.out}")
    else:
        check(args.decisions, args.labels, args.calibration)


if __name__ == "__main__":
    main()
