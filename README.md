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

A local web server that replicates the LeetGPU experience — browse challenges, edit CUDA code, and run tests locally with no submission limits.

### Setup

```bash
cd local_server
uv sync
```

### Usage

```bash
cd local_server
uv run python server.py
```

Open `http://127.0.0.1:5000` in your browser.

### Features

- Browse all 79 challenges grouped by difficulty
- Split-pane workspace: problem description (left) + code editor (right) + console output (bottom)
- **Run**: compile and test against functional tests
- **Submit**: run all tests including performance test; optionally save your solution on success
- Solutions saved to `challenges/<difficulty>/<name>/solution/solution_<timestamp>.cu`
- Code persists in browser localStorage between sessions

### Remote Access (Tailscale)

To access from another device on your Tailscale network:

```bash
# Default: binds to localhost + Tailscale IP automatically
uv run python server.py

# Or specify a host manually
uv run python server.py --host <your-tailscale-ip>
```

Make sure `tailscale0` is in the firewall trusted zone:

```bash
sudo firewall-cmd --zone=trusted --add-interface=tailscale0 --permanent
sudo firewall-cmd --reload
```

Then visit `http://<tailscale-ip>:5000` (use **http**, not https — Tailscale already encrypts traffic via WireGuard).

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- CUDA toolkit (`nvcc` on PATH)
- GPU with CUDA support

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## License

This problem set is licensed under [CC BY‑NC‑ND 4.0 license](LICENSE).

© 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.
