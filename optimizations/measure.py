"""Measure one command inside its GPU container; no scientific settings are changed."""

import argparse
import json
from pathlib import Path
import resource
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required")
    started = time.monotonic()
    samples = []
    process = subprocess.Popen(command)

    def stop(signum, _frame):
        process.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    while process.poll() is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            samples.extend(int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit())
        time.sleep(0.2)
    record = {
        "command": command, "exitCode": process.returncode,
        "wallSeconds": time.monotonic() - started,
        "peakHostKiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "peakDeviceMiB": max(samples) if samples else None,
        "gpuMeasurement": "sampled total visible-device usage; requires exclusive GPU",
        "gpuSamples": len(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
