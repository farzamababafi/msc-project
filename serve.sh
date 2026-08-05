#!/usr/bin/env bash
# Start SmolVLA websocket server with project defaults.
# Override any flag if needed, e.g.: ./serve.sh --port 6003
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/smolvla_server.py" "$@"
