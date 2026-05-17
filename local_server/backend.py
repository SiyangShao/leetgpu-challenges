"""
Execution backends for the LeetGPU local server.

LocalBackend: original behavior — nvcc + ctypes in a subprocess on this machine.
RemoteBackend: ships challenge.py + challenge_base.py + .cu source to a Runpod
Flash endpoint, which compiles and runs on a remote GPU.

Select via env var:  LEETGPU_BACKEND=local | remote   (default: local)
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHALLENGE_BASE_PATH = PROJECT_ROOT / "challenges" / "core" / "challenge_base.py"


class Backend(ABC):
    @abstractmethod
    def run(self, challenge_dir: str, code: str, test_type: str) -> Dict[str, Any]:
        """Compile + execute + compare.

        Returns the dict the Flask route serializes:
          {compilation: {success, stderr}, tests: [...], error?: str}
        """


# ---------------------------------------------------------------------------
# Local backend — original subprocess-based implementation.
# ---------------------------------------------------------------------------
def _compile_cuda(code: str, work_dir: str):
    src = os.path.join(work_dir, "solution.cu")
    so = os.path.join(work_dir, "solution.so")
    with open(src, "w") as f:
        f.write(code)
    result = subprocess.run(
        ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-o", so, src],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return False, result.stderr, None
    return True, result.stderr, so


def _execute_in_subprocess(challenge_dir: str, so_path: str, test_type: str, result_queue):
    try:
        import ctypes as ct

        import torch

        challenges_root = str(Path(challenge_dir).parent.parent)
        if challenges_root not in sys.path:
            sys.path.insert(0, challenges_root)

        challenge_py = os.path.join(challenge_dir, "challenge.py")
        spec = importlib.util.spec_from_file_location("challenge_mod", challenge_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        challenge_obj = mod.Challenge()

        lib = ct.CDLL(so_path)
        sig = challenge_obj.get_solve_signature()

        if test_type == "functional":
            test_cases = challenge_obj.generate_functional_test()
        elif test_type == "performance":
            test_cases = [challenge_obj.generate_performance_test()]
        elif test_type == "example":
            test_cases = [challenge_obj.generate_example_test()]
        else:
            test_cases = challenge_obj.generate_functional_test()
            test_cases.append(challenge_obj.generate_performance_test())

        results = []
        for i, test_case in enumerate(test_cases):
            ref_case = {}
            for param_name in sig:
                val = test_case[param_name]
                if isinstance(val, torch.Tensor):
                    ref_case[param_name] = val.clone()
                else:
                    ref_case[param_name] = val

            try:
                argtypes = []
                call_args = []
                for param_name, (ctype, _direction) in sig.items():
                    value = test_case[param_name]
                    if issubclass(ctype, ct._Pointer):
                        ptr = ct.cast(ct.c_void_p(value.data_ptr()), ctype)
                        argtypes.append(ctype)
                        call_args.append(ptr)
                    else:
                        argtypes.append(ctype)
                        call_args.append(ctype(int(value)))

                solve_func = lib.solve
                solve_func.argtypes = argtypes
                solve_func.restype = None

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                solve_func(*call_args)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                user_time_ms = (t1 - t0) * 1000

                challenge_obj.reference_impl(**ref_case)

                passed = True
                error_msg = None
                for param_name, (_ctype, direction) in sig.items():
                    if direction in ("out", "inout") and isinstance(
                        test_case[param_name], torch.Tensor
                    ):
                        user_t = test_case[param_name]
                        ref_t = ref_case[param_name]
                        if not torch.allclose(
                            user_t, ref_t, atol=challenge_obj.atol, rtol=challenge_obj.rtol
                        ):
                            diff = (user_t.float() - ref_t.float()).abs()
                            max_diff = diff.max().item()
                            max_idx = diff.argmax().item()
                            passed = False
                            error_msg = (
                                f"Mismatch in '{param_name}': "
                                f"max abs diff = {max_diff:.6e} at flat index {max_idx}, "
                                f"expected {ref_t.flatten()[max_idx].item():.6f}, "
                                f"got {user_t.flatten()[max_idx].item():.6f}"
                            )
                            break

                is_perf = test_type == "all" and i == len(test_cases) - 1
                results.append(
                    {
                        "index": i + 1,
                        "passed": passed,
                        "time_ms": round(user_time_ms, 3),
                        "error": error_msg,
                        "is_performance": test_type == "performance" or is_perf,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "index": i + 1,
                        "passed": False,
                        "time_ms": 0,
                        "error": str(e),
                        "is_performance": False,
                    }
                )

        result_queue.put({"success": True, "results": results})
    except Exception:
        import traceback

        result_queue.put({"success": False, "error": traceback.format_exc()})


class LocalBackend(Backend):
    """Runs nvcc + ctypes locally in a spawned subprocess (crash isolation)."""

    def run(self, challenge_dir: str, code: str, test_type: str) -> Dict[str, Any]:
        work_dir = tempfile.mkdtemp(prefix="leetgpu_")
        ok, stderr, so_path = _compile_cuda(code, work_dir)
        if not ok:
            return {"compilation": {"success": False, "stderr": stderr}, "tests": []}

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(
            target=_execute_in_subprocess, args=(challenge_dir, so_path, test_type, q)
        )
        p.start()
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            p.join(5)
            return {
                "compilation": {"success": True, "stderr": stderr},
                "tests": [],
                "error": "Execution timed out (120s)",
            }
        if q.empty():
            return {
                "compilation": {"success": True, "stderr": stderr},
                "tests": [],
                "error": "Process crashed (possible segfault in your CUDA code)",
            }
        outcome = q.get()
        if not outcome["success"]:
            return {
                "compilation": {"success": True, "stderr": stderr},
                "tests": [],
                "error": outcome["error"],
            }
        return {
            "compilation": {"success": True, "stderr": stderr},
            "tests": outcome["results"],
        }


# ---------------------------------------------------------------------------
# Remote backend — Runpod Flash.
# ---------------------------------------------------------------------------
class RemoteBackend(Backend):
    """Sends source code to a Runpod Flash @Endpoint; remote compiles & runs."""

    def __init__(self):
        from flash_worker import run_solution  # registers the @Endpoint

        self._endpoint = run_solution

    def run(self, challenge_dir: str, code: str, test_type: str) -> Dict[str, Any]:
        challenge_py = Path(challenge_dir) / "challenge.py"
        challenge_src = challenge_py.read_text()
        base_src = CHALLENGE_BASE_PATH.read_text()

        coro = self._endpoint(
            challenge_src=challenge_src,
            base_src=base_src,
            solution_cu=code,
            test_type=test_type,
        )
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
def get_backend() -> Backend:
    name = os.environ.get("LEETGPU_BACKEND", "local").lower()
    if name == "remote":
        return RemoteBackend()
    if name == "local":
        return LocalBackend()
    raise ValueError(f"Unknown LEETGPU_BACKEND={name!r} (expected 'local' or 'remote')")
