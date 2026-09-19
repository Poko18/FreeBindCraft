#!/usr/bin/env bash
# Called inside a fresh micromamba environment by Dockerfile.uplift.
set -euo pipefail
kit="${1:?Path to the checksum-verified optimization kit tree}"
micromamba install -y -n base -f "$kit/colabdesign/environment/conda-linux-64.lock"
micromamba clean -a -y
python -m pip install --no-cache-dir --no-deps \
  "$kit/colabdesign/stock/colabdesign-e31a56fe.tar.gz" 'freesasa==2.2.1'
python -m pip install --no-cache-dir --no-deps -e "$kit/common/opt_core" -e "$kit/colabdesign/opt"
python "$kit/colabdesign/stock/check_pins.py" --no-gpu
python -c 'import importlib.util, colabdesign, openmm, pdbfixer, freesasa; assert importlib.util.find_spec("pyrosetta") is None'
