"""
Runpod Flash worker — compiles user CUDA + executes against challenge tests on a remote GPU.

Deploy:
    flash login
    flash run --auto-provision    # local dev, warms a worker
    flash deploy                  # ship to production

The @Endpoint function receives source code as strings (no need to redeploy when
challenges change). Returns the same shape that server.py's _handle_execution
expects: {compilation: {success, stderr}, tests: [...], error?: str}.
"""

from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="leetgpu-runner",
    gpu=GpuGroup.AMPERE_16,
    workers=(0, 2),
    idle_timeout=30,
    dependencies=["torch", "numpy<2"],
    system_dependencies=["nvidia-cuda-toolkit"],
)
async def run_solution(
    challenge_src: str,
    base_src: str,
    solution_cu: str,
    test_type: str = "functional",
) -> dict:
    import ctypes as ct
    import importlib.util
    import os
    import subprocess
    import sys
    import tempfile
    import time
    import traceback

    try:
        import torch
    except ImportError as e:
        return {"error": f"torch import failed on worker: {e}"}

    work_dir = tempfile.mkdtemp(prefix="leetgpu_")

    core_dir = os.path.join(work_dir, "core")
    os.makedirs(core_dir, exist_ok=True)
    with open(os.path.join(core_dir, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(core_dir, "challenge_base.py"), "w") as f:
        f.write(base_src)

    challenge_py = os.path.join(work_dir, "challenge.py")
    with open(challenge_py, "w") as f:
        f.write(challenge_src)

    cu_path = os.path.join(work_dir, "solution.cu")
    so_path = os.path.join(work_dir, "solution.so")
    with open(cu_path, "w") as f:
        f.write(solution_cu)

    compile_result = subprocess.run(
        ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-o", so_path, cu_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if compile_result.returncode != 0:
        return {
            "compilation": {"success": False, "stderr": compile_result.stderr},
            "tests": [],
        }

    try:
        if work_dir not in sys.path:
            sys.path.insert(0, work_dir)

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

        return {
            "compilation": {"success": True, "stderr": compile_result.stderr},
            "tests": results,
        }
    except Exception:
        return {
            "compilation": {"success": True, "stderr": compile_result.stderr},
            "tests": [],
            "error": traceback.format_exc(),
        }
