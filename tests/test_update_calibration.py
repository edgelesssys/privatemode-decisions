"""scripts/update_calibration.py rewrites the generated block faithfully."""

import json
import re
import subprocess
import sys
from pathlib import Path

from decisions import calibration

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "update_calibration.py"
SOURCE = ROOT / "decisions" / "calibration.py"


def run(constants: dict, target: Path, tmp_path: Path) -> None:
    path = tmp_path / "constants.json"
    path.write_text(json.dumps(constants))
    subprocess.run([sys.executable, str(SCRIPT), str(path), "--target", str(target)],
                   check=True, capture_output=True)


def current_constants(model: str) -> dict:
    sources = dict(re.findall(r"^# source \((.+?)\): (.*)$", SOURCE.read_text(), re.M))
    return {"model": model, "formula": list(calibration.FORMULAS[model]),
            "family": calibration.FAMILY_TEMPERATURES[model], "source": sources[model]}


def test_regenerating_the_current_values_changes_nothing(tmp_path):
    target = tmp_path / "calibration.py"
    target.write_text(SOURCE.read_text())
    for model in calibration.FORMULAS:
        run(current_constants(model), target, tmp_path)
    assert target.read_text() == SOURCE.read_text()


def test_a_new_model_is_added_and_the_others_kept(tmp_path):
    target = tmp_path / "calibration.py"
    target.write_text(SOURCE.read_text())
    run({"model": "new-model", "alias": "new-latest", "formula": [1.0, -0.1],
         "family": {"topic": 2.0}, "source": "a test"}, target, tmp_path)
    text = target.read_text()
    assert '"new-model": (1.0, -0.1),' in text
    assert '"new-latest": "new-model",' in text
    assert "# source (new-model): a test" in text
    for model in calibration.FORMULAS:
        assert f'"{model}": {calibration.FORMULAS[model]},' in text
