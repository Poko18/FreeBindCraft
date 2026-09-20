"""Small, opt-in controls shared by the launcher and relaxation children."""

import hashlib
import os
from pathlib import Path


def strict():
    return os.environ.get("FREEBINDCRAFT_STRICT") == "1"


def relaxation_seed():
    value = os.environ.get("FREEBINDCRAFT_RELAX_SEED")
    return int(value) if value is not None else None


def missing_atoms(fixer):
    seed = relaxation_seed()
    return fixer.addMissingAtoms(**({"seed": seed} if seed is not None else {}))


def seed_integrator(integrator):
    seed = relaxation_seed()
    if seed is not None:
        integrator.setRandomNumberSeed(seed)


def relaxation_environment(pdb_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("COLABDESIGN_OPT")}
    if "FREEBINDCRAFT_SEED" in env:
        # Logical name, not the work/output directory or retry number.
        material = f'{env["FREEBINDCRAFT_SEED"]}:{Path(pdb_path).name}'.encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % 2147483646 + 1
        env["FREEBINDCRAFT_RELAX_SEED"] = str(seed)
    return env


def seed_relaxation():
    seed = relaxation_seed()
    if seed is not None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)


def require_backend(ok, name):
    if strict() and not ok:
        raise RuntimeError(f"Required FreeBindCraft backend failed: {name}")
