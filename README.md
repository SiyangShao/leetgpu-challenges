# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases, and starter templates for multiple GPU programming frameworks.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for each supported framework

## Local Server

A local web server that replicates the LeetGPU experience — browse challenges, edit CUDA code, and run tests with no submission limits.

The server has **two execution backends**, picked at startup:

| Backend  | When to use                  | How CUDA runs                                       |
|----------|------------------------------|-----------------------------------------------------|
| `local`  | You have a local NVIDIA GPU  | `nvcc` + `ctypes` on your machine (subprocess)      |
| `remote` | Mac / any host without CUDA  | Compiled & executed on [Runpod Serverless](https://runpod.io) via [`runpod-flash`](https://github.com/runpod/flash) |

Same UI, same tests, same Run/Submit flow — only where the compile + GPU execution happens differs.

### Quick start — Mac (no local GPU)

One command:

```bash
bash local_server/start-mac.sh
```

This script `uv sync`s the deps, runs `flash login` on first use, `flash deploy`s the worker if not yet deployed, and starts the UI at **http://127.0.0.1:5050** (port 5050 to avoid macOS AirPlay clash on 5000).

- **First Run click**: ~60–90s cold start (Runpod provisions a worker)
- **Subsequent clicks**: ~1s after deploy + 0.1–5s per test case
- Worker auto-scales to zero after 30s idle (≈ $0.02 wasted per idle window at A4000 prices)
- Re-deploy only needed when you change `local_server/flash_worker.py`: `cd local_server && uv run flash deploy`

#### Caveats

- Remote GPU is `GpuGroup.AMPERE_16` (RTX A4000-class), **not** the Tesla T4 used by LeetGPU's grader. Use this backend for **correctness only**; verify performance via `scripts/run_challenge.py`.
- Each invocation ships `challenge.py` source as payload — editing a challenge requires no redeploy.
- `start-mac.sh` accepts pass-through flags: `bash local_server/start-mac.sh --port 8080`.

### Quick start — Linux with local CUDA

```bash
cd local_server
uv sync
uv run python server.py
```

Open `http://127.0.0.1:5000`. Requires `nvcc` on PATH and a CUDA-capable GPU.

### Features

- Browse all challenges grouped by difficulty
- Split-pane workspace: problem description (left) + code editor (right) + console output (bottom)
- **Run**: compile + functional tests only
- **Submit**: full functional + performance test pass; on full success prompts to save your solution to `challenges/<difficulty>/<name>/solution/solution_<timestamp>.cu` (the `solution/` dir is git-ignored)
- Code persists in browser localStorage between sessions
- Submission history kept per-challenge in localStorage

### Remote Access (Tailscale, local backend only)

To access from another device on your Tailscale network:

```bash
uv run python server.py                  # auto-binds to localhost + Tailscale IP
uv run python server.py --host <ip>      # or a specific interface
```

Open `tailscale0` in the firewall trusted zone:

```bash
sudo firewall-cmd --zone=trusted --add-interface=tailscale0 --permanent
sudo firewall-cmd --reload
```

Visit `http://<tailscale-ip>:5000` (use **http**, not https — Tailscale already encrypts via WireGuard).

### Switching backends explicitly

```bash
LEETGPU_BACKEND=remote uv run python server.py     # remote
LEETGPU_BACKEND=local  uv run python server.py     # local (default)
uv run python server.py --remote                   # shorthand for remote
```

### Requirements

| Backend  | Requirements |
|----------|--------------|
| `local`  | Python 3.10+, [uv](https://docs.astral.sh/uv/), `nvcc` on PATH, CUDA GPU |
| `remote` | Python 3.10+, [uv](https://docs.astral.sh/uv/), [Runpod account](https://console.runpod.io) with API balance |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## License

This problem set is licensed under [CC BY‑NC‑ND 4.0 license](LICENSE).

© 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.
