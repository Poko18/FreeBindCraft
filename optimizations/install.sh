#!/usr/bin/env bash
# Called inside a fresh micromamba environment by Dockerfile.uplift.
set -euo pipefail
kit="${1:?Path to the checksum-verified optimization kit tree}"
micromamba install -y -n base -f "$kit/colabdesign/environment/conda-linux-64.lock"
micromamba clean -a -y
# The upstream sdist omits model data when built as a wheel. An editable install
# retains the checksum-verified MPNN weights and AF template head from the kit.
mkdir -p /opt/colabdesign-source
tar -xzf "$kit/colabdesign/stock/colabdesign-e31a56fe.tar.gz" \
  -C /opt/colabdesign-source --strip-components=1
python -m pip install --no-cache-dir --no-deps \
  -e /opt/colabdesign-source 'freesasa==2.2.1'
python -m pip install --no-cache-dir --no-deps -e "$kit/common/opt_core" -e "$kit/colabdesign/opt"
python "$kit/colabdesign/stock/check_pins.py" --no-gpu
python -c 'import importlib.util, colabdesign, openmm, pdbfixer, freesasa; assert importlib.util.find_spec("pyrosetta") is None'
