"""Supervise the complete PyRosetta-free pipeline with pinned ColabDesign modes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIT_REVISION = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"


def uint32(value):
    result = int(value)
    if not 0 <= result < 2**32:
        raise argparse.ArgumentTypeError("Seed must be an unsigned 32-bit integer")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "exact", "fast"), default="off")
    parser.add_argument("--seed", type=uint32)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def inputs(command):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-s", "--settings", required=True, type=Path)
    parser.add_argument("-a", "--advanced", default=ROOT / "settings_advanced/default_4stage_multimer.json", type=Path)
    parser.add_argument("-f", "--filters", default=ROOT / "settings_filters/default_filters.json", type=Path)
    parser.add_argument("--no-pyrosetta", action="store_true")
    args, _ = parser.parse_known_args(command)
    if not args.no_pyrosetta:
        parser.error("This launcher requires --no-pyrosetta")
    settings = json.loads(args.settings.read_text())
    advanced = json.loads(args.advanced.read_text())
    output = Path(settings["design_path"]).resolve()
    return output, advanced


def source_digest():
    digest = hashlib.sha256()
    for path in sorted([ROOT / "bindcraft.py", *ROOT.glob("functions/*.py"), *ROOT.glob("optimizations/*.py")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(mode, seed, advanced):
    env = {k: v for k, v in os.environ.items() if not k.startswith("COLABDESIGN_OPT")}
    env.update(PYTHONHASHSEED="0", FREEBINDCRAFT_SEED=str(seed), FREEBINDCRAFT_STRICT="1")
    # Include external loss callback source and settings absent from the kit's key.
    key = hashlib.sha256(json.dumps([source_digest(), advanced], sort_keys=True).encode()).hexdigest()
    root = Path(env.get("MODEL_OPT_JIT_ROOT", env.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))))
    env["XDG_CACHE_HOME"] = str(root / "freebindcraft" / key)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    if mode != "off":
        env["COLABDESIGN_OPT"] = mode
    return env


def activation_verdict(mode, lines):
    if mode == "off":
        bad = any("[colabdesign-opt] LEVER " in line for line in lines)
        return {"partial": {"off": "unexpected optimization activation"} if bad else {}, "gated": {}, "skipped": {}}
    from colabdesign_opt import evidence, modes

    verdict = evidence.verdict(evidence.classify(lines, modes.resolve(mode).levers))
    if not any(f"ACTIVE mode={mode} " in line for line in lines):
        verdict["partial"]["activation"] = "missing ACTIVE line for selected mode"
    if any("NOT ACTIVE" in line for line in lines):
        verdict["partial"]["activation"] = "runtime refused activation"
    return verdict


def child(command, seed):
    # ColabDesign assigns compiler flags before JAX creates its backend.
    import colabdesign  # noqa: F401
    import jax
    import numpy as np
    import freesasa  # noqa: F401

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("FreeBindCraft requires a JAX GPU; refusing CPU fallback")
    random.seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "bindcraft.py"), *command]
    runpy.run_path(str(ROOT / "bindcraft.py"), run_name="__main__")


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    output, advanced = inputs(command)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    if args.child:
        child(command, seed)
        return 0

    output.mkdir(parents=True, exist_ok=True)
    record = {
        "tool": "freebindcraft", "mode": args.mode, "seed": seed,
        "kitRevision": KIT_REVISION, "adapterRevision": 1,
        "sourceRevision": os.environ.get("FREEBINDCRAFT_REVISION", "local"),
        "sourceDigest": source_digest(), "command": command, "complete": False,
        "dependencies": {}, "backendEvidence": [],
    }
    for name in ("jax", "jaxlib", "colabdesign", "openmm", "pdbfixer", "freesasa"):
        try:
            record["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record["dependencies"][name] = None

    def save():
        temporary = output / "execution.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output / "execution.json")

    save()
    lines = []
    code = 1
    try:
        with (output / "optimization.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode,
                 "--seed", str(seed), "--child", "--", *command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=environment(args.mode, seed, advanced), start_new_session=True,
            )

            def stop(signum, _frame):
                record["canceled"] = True
                save()
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

            previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if "[colabdesign-opt]" in line:
                        lines.append(line)
                    if "[freebindcraft-backend]" in line:
                        record["backendEvidence"].append(line.strip())
                code = process.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        if code == 0 and not record.get("canceled"):
            record["activation"] = activation_verdict(args.mode, lines)
            if record["activation"]["partial"]:
                code = 3
        if record.get("canceled") and code == 0:
            code = 143
    except Exception as exc:
        code = 1
        record["error"] = str(exc)
        raise
    finally:
        code = code if code >= 0 else 128 - code
        record.update(exitCode=code, complete=code == 0 and not record.get("canceled", False))
        save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
