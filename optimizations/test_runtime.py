"""Run without scientific dependencies: python -m unittest discover -s optimizations."""

import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run
import runtime


class RuntimeTests(unittest.TestCase):
    def test_colabdesign_precedes_gpu_query_and_no_pyrosetta_is_imported(self):
        import builtins

        events = []
        original_import = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "colabdesign":
                events.append("colabdesign")
            if name == "pyrosetta":
                raise AssertionError("PyRosetta must not be imported")
            return original_import(name, *args, **kwargs)

        def devices():
            self.assertIn("colabdesign", events)
            return [SimpleNamespace(platform="gpu")]

        modules = {"colabdesign": SimpleNamespace(), "jax": SimpleNamespace(devices=devices),
                   "numpy": SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
                   "freesasa": SimpleNamespace()}
        with patch.dict(sys.modules, modules), patch.object(builtins, "__import__", importing), \
             patch.object(run.runpy, "run_path") as launch, patch.object(sys, "argv", []), \
             patch.object(sys, "path", list(sys.path)):
            run.child(["--no-pyrosetta"], 42)
            launch.assert_called_once_with(str(run.ROOT / "bindcraft.py"), run_name="__main__")

    def test_seed_and_helper_environment(self):
        with patch.dict(os.environ, {"FREEBINDCRAFT_SEED": "42", "COLABDESIGN_OPT": "fast",
                                     "COLABDESIGN_OPT_LOWERCACHE": "relower",
                                     "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            first = runtime.relaxation_environment("/one/design.pdb")
            second = runtime.relaxation_environment("/two/design.pdb")
            self.assertEqual(first, second)
            self.assertFalse(any(k.startswith("COLABDESIGN_OPT") for k in first))
            self.assertEqual(first["CUDA_VISIBLE_DEVICES"], "0")
            self.assertNotEqual(first["FREEBINDCRAFT_RELAX_SEED"],
                                runtime.relaxation_environment("/one/other.pdb")["FREEBINDCRAFT_RELAX_SEED"])

    def test_legacy_helpers_remain_unseeded_and_strict_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(runtime.relaxation_seed())
            self.assertNotIn("FREEBINDCRAFT_RELAX_SEED", runtime.relaxation_environment("x"))
            runtime.require_backend(False, "OpenMM")
        with patch.dict(os.environ, {"FREEBINDCRAFT_STRICT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "OpenMM"):
                runtime.require_backend(False, "OpenMM")

    def test_modes_cannot_inherit_activation_and_settings_separate_caches(self):
        with patch.dict(os.environ, {"COLABDESIGN_OPT": "fast", "COLABDESIGN_OPT_OTHER": "x",
                                     "MODEL_OPT_JIT_ROOT": "/jit"}, clear=True):
            off = run.environment("off", 42, {"weights_rg": 1})
            self.assertFalse(any(k.startswith("COLABDESIGN_OPT") for k in off))
            exact = run.environment("exact", 42, {"weights_rg": 1})
            self.assertEqual(exact["COLABDESIGN_OPT"], "exact")
            self.assertNotEqual(exact["XDG_CACHE_HOME"],
                                run.environment("exact", 42, {"weights_rg": 2})["XDG_CACHE_HOME"])

    def test_uint32_and_off_evidence(self):
        self.assertEqual(run.uint32("4294967295"), 4294967295)
        for value in ("-1", "4294967296"):
            with self.assertRaises(Exception):
                run.uint32(value)
        self.assertTrue(run.activation_verdict("off", ["[colabdesign-opt] LEVER name=compilecache"])["partial"])

    def test_supervision_preserves_arguments_and_cancellation(self):
        for canceled in (False, True):
            with self.subTest(canceled=canceled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "settings.json").write_text(json.dumps({"design_path": tmp}))
                (root / "advanced.json").write_text("{}")
                native = ["-s", str(root / "settings.json"), "-a", str(root / "advanced.json"),
                          "--no-pyrosetta", "--rank-by", "ipSAE", "--no-plots"]

                class Process:
                    pid = 1234

                    @property
                    def stdout(self):
                        if canceled:
                            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                        return io.StringIO("pipeline complete\n")

                    def wait(self):
                        return 0

                with patch.object(run.subprocess, "Popen", return_value=Process()) as popen, \
                     patch.object(run.os, "killpg") as kill:
                    code = run.main(["--mode", "off", "--seed", "42", "--", *native])
                record = json.loads((root / "execution.json").read_text())
                self.assertEqual(record["complete"], not canceled)
                self.assertEqual(code, 143 if canceled else 0)
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index("--") + 1:], native)
                if canceled:
                    kill.assert_called_once_with(1234, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
