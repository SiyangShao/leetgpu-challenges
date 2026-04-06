#!/usr/bin/env python3
"""
Local LeetGPU server — browse challenges, edit CUDA, run tests locally.

Usage:
    cd /path/to/leetgpu-challenges
    python local_server/server.py
    # Open http://localhost:5000
"""

import ctypes
import importlib.util
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_ROOT = PROJECT_ROOT / "challenges"

# Ensure challenge imports work (from core.challenge_base import ...)
if str(CHALLENGES_ROOT) not in sys.path:
    sys.path.insert(0, str(CHALLENGES_ROOT))

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))

# ---------------------------------------------------------------------------
# Challenge registry
# ---------------------------------------------------------------------------
REGISTRY = {}  # slug -> challenge info dict
SLUG_ORDER = []  # ordered list of slugs


def _make_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug).strip("-")
    return slug


def discover_challenges():
    global REGISTRY, SLUG_ORDER
    challenges_by_difficulty = {"easy": [], "medium": [], "hard": []}

    for difficulty in ("easy", "medium", "hard"):
        diff_dir = CHALLENGES_ROOT / difficulty
        if not diff_dir.is_dir():
            continue
        for challenge_dir in sorted(diff_dir.iterdir()):
            if not challenge_dir.is_dir():
                continue
            challenge_py = challenge_dir / "challenge.py"
            challenge_html = challenge_dir / "challenge.html"
            starter_cu = challenge_dir / "starter" / "starter.cu"
            if not challenge_py.exists():
                continue

            # Parse number from directory name
            match = re.match(r"(\d+)_(.+)", challenge_dir.name)
            if not match:
                continue
            number = int(match.group(1))

            # Dynamic import
            try:
                spec = importlib.util.spec_from_file_location(
                    f"challenge_{difficulty}_{number}", challenge_py
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                challenge_obj = mod.Challenge()
            except Exception as e:
                print(f"  SKIP {challenge_dir.name}: {e}")
                continue

            slug = _make_slug(challenge_obj.name)
            desc_html = challenge_html.read_text() if challenge_html.exists() else ""
            starter = starter_cu.read_text() if starter_cu.exists() else ""

            entry = {
                "slug": slug,
                "dir_name": challenge_dir.name,
                "dir_path": str(challenge_dir),
                "number": number,
                "display_name": challenge_obj.name,
                "difficulty": difficulty,
                "challenge_obj": challenge_obj,
                "description_html": desc_html,
                "starter_code": starter,
            }
            REGISTRY[slug] = entry
            challenges_by_difficulty[difficulty].append(entry)

    # Build ordered list
    SLUG_ORDER.clear()
    for diff in ("easy", "medium", "hard"):
        for entry in sorted(challenges_by_difficulty[diff], key=lambda e: e["number"]):
            SLUG_ORDER.append(entry)

    print(f"Loaded {len(REGISTRY)} challenges")
    return challenges_by_difficulty


# ---------------------------------------------------------------------------
# CUDA compilation
# ---------------------------------------------------------------------------
def compile_cuda(code: str, work_dir: str) -> tuple:
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


# ---------------------------------------------------------------------------
# Test execution (runs in subprocess for crash isolation)
# ---------------------------------------------------------------------------
def _execute_in_subprocess(challenge_dir: str, so_path: str, test_type: str, result_queue):
    """Run inside a child process so segfaults don't kill the server."""
    try:
        import ctypes as ct

        import torch

        # Re-import challenge module
        challenges_root = str(Path(challenge_dir).parent.parent)
        if challenges_root not in sys.path:
            sys.path.insert(0, challenges_root)

        challenge_py = os.path.join(challenge_dir, "challenge.py")
        spec = importlib.util.spec_from_file_location("challenge_mod", challenge_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        challenge_obj = mod.Challenge()

        # Load compiled library
        lib = ct.CDLL(so_path)

        sig = challenge_obj.get_solve_signature()

        # Generate test cases
        if test_type == "functional":
            test_cases = challenge_obj.generate_functional_test()
        elif test_type == "performance":
            test_cases = [challenge_obj.generate_performance_test()]
        elif test_type == "example":
            test_cases = [challenge_obj.generate_example_test()]
        else:
            test_cases = challenge_obj.generate_functional_test()
            perf = challenge_obj.generate_performance_test()
            test_cases.append(perf)

        results = []
        for i, test_case in enumerate(test_cases):
            # Clone tensors for reference comparison
            ref_case = {}
            for param_name in sig:
                val = test_case[param_name]
                if isinstance(val, torch.Tensor):
                    ref_case[param_name] = val.clone()
                else:
                    ref_case[param_name] = val

            try:
                # Build ctypes arguments
                argtypes = []
                call_args = []
                for param_name, (ctype, direction) in sig.items():
                    value = test_case[param_name]
                    if issubclass(ctype, ct._Pointer):
                        # Pointer type — value is a torch.Tensor
                        ptr = ct.cast(ct.c_void_p(value.data_ptr()), ctype)
                        argtypes.append(ctype)
                        call_args.append(ptr)
                    else:
                        # Scalar type
                        argtypes.append(ctype)
                        call_args.append(ctype(int(value)))

                solve_func = lib.solve
                solve_func.argtypes = argtypes
                solve_func.restype = None

                # Run user's solve with timing
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                solve_func(*call_args)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                user_time_ms = (t1 - t0) * 1000

                # Run reference
                challenge_obj.reference_impl(**ref_case)

                # Compare output tensors
                passed = True
                error_msg = None
                for param_name, (ctype, direction) in sig.items():
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
    except Exception as e:
        import traceback

        result_queue.put({"success": False, "error": traceback.format_exc()})


def execute_tests(challenge_dir: str, so_path: str, test_type: str, timeout: int = 120):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_execute_in_subprocess, args=(challenge_dir, so_path, test_type, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return {"success": False, "error": f"Execution timed out ({timeout}s)"}
    if q.empty():
        return {"success": False, "error": "Process crashed (possible segfault in your CUDA code)"}
    return q.get()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    grouped = {"easy": [], "medium": [], "hard": []}
    for entry in SLUG_ORDER:
        grouped[entry["difficulty"]].append(entry)
    return render_template("index.html", grouped=grouped)


@app.route("/challenges/<slug>")
def challenge_page(slug):
    entry = REGISTRY.get(slug)
    if not entry:
        return "Challenge not found", 404
    # Find prev/next for navigation
    idx = next((i for i, e in enumerate(SLUG_ORDER) if e["slug"] == slug), -1)
    prev_entry = SLUG_ORDER[idx - 1] if idx > 0 else None
    next_entry = SLUG_ORDER[idx + 1] if idx < len(SLUG_ORDER) - 1 else None
    return render_template(
        "challenge.html", entry=entry, prev_entry=prev_entry, next_entry=next_entry
    )


@app.route("/api/challenges/<slug>/run", methods=["POST"])
def api_run(slug):
    return _handle_execution(slug, "functional")


@app.route("/api/challenges/<slug>/submit", methods=["POST"])
def api_submit(slug):
    return _handle_execution(slug, "all")



def _handle_execution(slug, test_type):
    entry = REGISTRY.get(slug)
    if not entry:
        return jsonify({"error": "Challenge not found"}), 404

    data = request.get_json()
    code = data.get("code", "")
    if not code.strip():
        return jsonify({"error": "No code provided"}), 400

    # Compile
    work_dir = tempfile.mkdtemp(prefix="leetgpu_")
    try:
        ok, stderr, so_path = compile_cuda(code, work_dir)
        if not ok:
            return jsonify(
                {
                    "compilation": {"success": False, "stderr": stderr},
                    "tests": [],
                }
            )

        # Execute tests in subprocess
        result = execute_tests(entry["dir_path"], so_path, test_type)

        if not result["success"]:
            return jsonify(
                {
                    "compilation": {"success": True, "stderr": stderr},
                    "tests": [],
                    "error": result["error"],
                }
            )

        tests = result["results"]
        all_passed = all(t["passed"] for t in tests)

        return jsonify(
            {
                "compilation": {"success": True, "stderr": stderr},
                "tests": tests,
                "all_passed": test_type == "all" and all_passed,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_tailscale_ip():
    """Get the Tailscale IPv4 address, or None if unavailable."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local LeetGPU server")
    parser.add_argument("--host", default=None, help="Bind address (default: 127.0.0.1 + tailscale)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--unsafe", action="store_true", help="Bind to 0.0.0.0 (all interfaces)")
    args = parser.parse_args()

    print("Discovering challenges...")
    discover_challenges()

    if args.unsafe:
        print(f"WARNING: Binding to 0.0.0.0:{args.port} (all interfaces)")
        app.run(host="0.0.0.0", port=args.port, debug=False)
    elif args.host:
        print(f"Starting server at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        # Default: localhost + tailscale only
        from threading import Thread

        ts_ip = _get_tailscale_ip()
        print(f"Starting server at http://127.0.0.1:{args.port}")
        if ts_ip:
            print(f"  Also listening on http://{ts_ip}:{args.port} (tailscale)")
            Thread(
                target=lambda: app.run(host=ts_ip, port=args.port, debug=False),
                daemon=True,
            ).start()
        else:
            print("  Tailscale not detected — localhost only")
        app.run(host="127.0.0.1", port=args.port, debug=False)
