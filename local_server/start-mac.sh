#!/usr/bin/env bash
# One-liner Mac launcher: sync deps + ensure flash login + start the web UI
# with the Runpod Flash remote backend.
#
# Usage:
#     bash local_server/start-mac.sh
#
# Or make it executable once:  chmod +x local_server/start-mac.sh
#                              ./local_server/start-mac.sh
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not installed. Install with:  brew install uv" >&2
    exit 1
fi

echo "==> Syncing dependencies (with [remote] extra)..."
uv sync --extra remote

# Auth check — runpod-flash writes ~/.runpod/config.toml on `flash login`,
# or honours RUNPOD_API_KEY.
if [[ -z "${RUNPOD_API_KEY:-}" ]] && [[ ! -f "$HOME/.runpod/config.toml" ]]; then
    echo "==> Not logged in to Runpod. Launching 'flash login'..."
    uv run flash login
fi

# Deploy check — `await endpoint(...)` requires the endpoint to be live on Runpod.
# Re-deploy manually with `uv run flash deploy` after editing flash_worker.py.
APP_LIST=$(uv run flash app list 2>&1 || true)
if echo "$APP_LIST" | grep -q "no apps found" || ! echo "$APP_LIST" | grep -q "local_server"; then
    echo "==> Endpoint not deployed yet. Running 'flash deploy' (1-2 min)..."
    uv run flash deploy
fi

# macOS uses port 5000 for AirPlay Receiver — default to 5050 to avoid the clash.
# Override with:  bash start-mac.sh --port 1234
PORT=5050
for arg in "$@"; do
    if [[ "$arg" == "--port" ]] || [[ "$arg" == --port=* ]]; then
        PORT=""   # user supplied one; don't inject ours
        break
    fi
done

if [[ -n "$PORT" ]]; then
    echo "==> Starting LeetGPU server (remote backend) on http://127.0.0.1:$PORT"
    exec uv run python server.py --remote --port "$PORT" "$@"
else
    exec uv run python server.py --remote "$@"
fi
