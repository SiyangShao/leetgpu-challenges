#!/usr/bin/env python3
"""
Local LeetGPU server — browse challenges, edit CUDA, run tests locally.

Usage:
    cd /path/to/leetgpu-challenges
    python local_server/server.py
    # Open http://localhost:5000
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from backend import Backend, get_backend

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
# Backend (local nvcc / remote runpod-flash) — chosen via LEETGPU_BACKEND env
# ---------------------------------------------------------------------------
BACKEND: Backend | None = None  # set in main()


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


@app.route("/api/challenges/<slug>/save", methods=["POST"])
def api_save(slug):
    from datetime import datetime

    entry = REGISTRY.get(slug)
    if not entry:
        return jsonify({"error": "Challenge not found"}), 404

    data = request.get_json()
    code = data.get("code", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sol_dir = os.path.join(entry["dir_path"], "solution")
    os.makedirs(sol_dir, exist_ok=True)
    sol_file = os.path.join(sol_dir, f"solution_{ts}.cu")
    with open(sol_file, "w") as f:
        f.write(code)
    saved_path = os.path.relpath(sol_file, PROJECT_ROOT)
    return jsonify({"saved": saved_path})



def _handle_execution(slug, test_type):
    entry = REGISTRY.get(slug)
    if not entry:
        return jsonify({"error": "Challenge not found"}), 404

    data = request.get_json()
    code = data.get("code", "")
    if not code.strip():
        return jsonify({"error": "No code provided"}), 400

    try:
        result = BACKEND.run(entry["dir_path"], code, test_type)
        tests = result.get("tests", [])
        all_passed = bool(tests) and all(t["passed"] for t in tests)
        result["all_passed"] = test_type == "all" and all_passed
        return jsonify(result)
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
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run CUDA on a Runpod Flash GPU instead of local nvcc (sets LEETGPU_BACKEND=remote)",
    )
    args = parser.parse_args()

    if args.remote:
        os.environ["LEETGPU_BACKEND"] = "remote"

    BACKEND = get_backend()
    print(f"Backend: {BACKEND.__class__.__name__}")

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
