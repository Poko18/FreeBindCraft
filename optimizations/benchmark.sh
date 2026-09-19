#!/usr/bin/env bash
# Workstation-only measurement queue. Uses an already built image and read-only source/weights.
# bash optimizations/benchmark.sh STUDY_DIR IMAGE WEIGHTS_DIR [fast-cold fast-warm off-b off-c exact-cold exact-warm]
set -euo pipefail
study="${1:?study directory}"
image="${2:?image}"
weights="${3:?AF2 directory}"
shift 3
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# == 0 ]]; then set -- fast-cold fast-warm off-b off-c exact-cold exact-warm; fi
mkdir -p "$study/jit"
for case_name in "$@"; do
  mode="${case_name%%-*}"
  case "$mode" in off|exact|fast) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
  if [[ ! "$case_name" =~ ^[a-z0-9-]+$ ]]; then exit 2; fi
  work="$study/$case_name"
  if [[ -e "$work" ]]; then echo "Refusing to overwrite $work" >&2; exit 2; fi
  while true; do
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    [[ -z "$active" ]] && break
    echo "Waiting for idle GPU before $case_name"
    sleep 10
  done
  python3 "$source_dir/optimizations/prepare_fixture.py" "$work/input" --exercise-all-stages
  mkdir -p "$work/output"
  container="freebindcraft-study-$case_name"
  echo "Starting $case_name"
  set +e
  timeout --signal=TERM --kill-after=30s 1800 docker run --rm --pull never \
    --name "$container" --gpus device=0 \
    -v "$source_dir:/app:ro" -v "$work:/work" -v "$weights:/models/af2:ro" \
    -v "$study/jit:/jit" -e MODEL_OPT_JIT_ROOT=/jit "$image" \
    python optimizations/measure.py --output /work/measurement.json -- \
    python optimizations/run.py --mode "$mode" --seed 42 -- \
    -s /work/input/settings.json -a /work/input/advanced.json -f /work/input/filters.json \
    --no-pyrosetta --rank-by ipSAE --no-plots --no-animations --verbose \
    > "$work/run.log" 2>&1
  result=$?
  set -e
  if docker inspect "$container" >/dev/null 2>&1; then docker stop --time 20 "$container"; fi
  echo "$case_name exited $result"
  # A failed run needs investigation before the next measurement.
  if [[ $result != 0 ]]; then exit "$result"; fi
done
