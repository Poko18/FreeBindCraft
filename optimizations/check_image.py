"""Run the shipped helper binaries on a small PDL1 fixture during image build."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert importlib.util.find_spec("pyrosetta") is None
    from colabdesign import mk_mpnn_model

    # Exercise loading (not merely importing) the preset's packaged weights.
    mk_mpnn_model(model_name="v_48_020", weights="soluble")
    sys.path.insert(0, str(ROOT))
    from functions import pr_alternative_utils as scoring

    for contacts, error, passes in [({}, "No molecular dots generated", True),
                                   ({1: "A"}, "No molecular dots generated", False),
                                   ({}, "Invalid radii file", False)]:
        failure = subprocess.CalledProcessError(1, ["sc"], stderr=error)
        with patch.dict(os.environ, {"FREEBINDCRAFT_STRICT": "1"}), \
             patch.object(scoring, "hotspot_residues", return_value=contacts), \
             patch.object(subprocess, "run", side_effect=failure):
            try:
                value = scoring._calculate_shape_complementarity("complex.pdb")
            except RuntimeError:
                assert not passes
            else:
                assert passes and value == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = ROOT / "example/PDL1.pdb"
        subprocess.run([str(ROOT / "functions/dssp"), "-i", str(target), "-o", str(root / "target.dssp")], check=True)
        assert (root / "target.dssp").stat().st_size > 0
        subprocess.run([str(ROOT / "functions/FASPR"), "-i", str(target), "-o", str(root / "repacked.pdb")], cwd=ROOT / "functions", check=True)
        assert (root / "repacked.pdb").stat().st_size > 0
        # Split one structure into two chains solely to exercise the SC executable.
        lines = [line[:21] + ("B" if int(line[22:26]) > 60 else "A") + line[22:]
                 for line in target.read_text().splitlines(True) if line.startswith("ATOM")]
        (root / "complex.pdb").write_text("".join(lines) + "END\n")
        result = subprocess.run([str(ROOT / "functions/sc"), str(root / "complex.pdb"), "A", "B", "--json"], check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
        assert 0 <= float(value.get("sc", value.get("sc_value"))) <= 1


if __name__ == "__main__":
    main()
