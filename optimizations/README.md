# Optional ColabDesign acceleration

This runs the complete FreeBindCraft pipeline without PyRosetta. The environment
is pinned to Anthropic's `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` kit and
ColabDesign `e31a56fe1d9b4de25c8697f3a28b75892941cc72`. The original installer and
CLI remain available. The optional PyRosetta path is not supported by this launcher.

```sh
docker build -f Dockerfile.uplift -t freebindcraft-uplift .
docker run --rm --gpus all \
  -v /path/to/af2:/models/af2:ro -v /path/to/run:/work -v /path/to/jit:/jit \
  -e MODEL_OPT_JIT_ROOT=/jit freebindcraft-uplift \
  python optimizations/run.py --mode exact --seed 42 -- \
  -s /work/input/settings.json -f /work/input/filters.json \
  -a /work/input/advanced.json --no-pyrosetta --rank-by ipSAE
```

The AF2 mount must contain `params/*.npz` (multimer v3 and pTM weights). Weights
are not distributed in this image. Settings must use `/work/output` as
`design_path` and a container-visible target path. Download weights through
the original AlphaFold distribution and retain its license and checksums.

Modes: `off` (default) uses the same pinned environment without kit activation;
`exact` enables the kit's exact levers; `fast` additionally enables numerical
changes and is experimental. Exact means ColabDesign computations for identical
inputs, not identical final designs. Existing OpenMM GPU variability remains even
with fixed seeds. OpenMM changes coordinates, not amino-acid identities. Those
coordinate changes can alter interface metrics and which sequence positions are
kept fixed during subsequent MPNN redesign. Qualify repeated off and
off/exact ColabDesign outputs before relaxation on your GPU. No end-to-end equality
or speedup is implied by a mode name.

The launcher preserves the native pipeline and writes `execution.json` and
`optimization.log` under `design_path`. It rejects missing/partial activation,
CPU JAX fallback, missing scientific backends and failed relaxation. It fixes
the design seed and derives stable per-structure relaxation seeds in all modes.
OpenMM subprocesses do not inherit optimization activation. SIGINT/SIGTERM reach
the design process group, including its OpenMM helpers.

Exact and fast caches are keyed by the kit, stack, fork source, and advanced
settings. Use a private writable cache directory per user/organization. A first
run compiles; measure cold and warm runs separately. Keep identical seeds,
inputs, settings, hardware and backend choices when comparing modes.

Run dependency-free checks with `python -m unittest discover -s optimizations`.
The kit's Apache-2.0 LICENSE and NOTICE and its third-party notices are retained
under `/kit`; FreeBindCraft and its helper binaries retain their own licenses.
