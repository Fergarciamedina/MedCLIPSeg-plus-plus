#!/usr/bin/env bash
# Start the dashboard only on this computer. Nothing is uploaded or exposed
# to the network: both services bind exclusively to 127.0.0.1.
set -euo pipefail

dashboard_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$dashboard_dir/../.." && pwd)"
venv_python="$repo_dir/.venv/bin/python"
frontend_dir="$dashboard_dir/frontend"
backend_dir="$dashboard_dir/backend"

if [[ ! -x "$venv_python" ]]; then
  echo "Creating the local Python environment…"
  python3 -m venv "$repo_dir/.venv"
fi

if ! "$venv_python" -c 'import fastapi, uvicorn, multipart, numpy, PIL, torch, faiss, pandas, matplotlib' 2>/dev/null; then
  echo "Missing local Python dependencies. Run this once while connected to your package mirror:"
  echo "  $venv_python -m pip install -r $backend_dir/requirements.txt"
  exit 1
fi

if [[ ! -x "$frontend_dir/node_modules/.bin/vite" ]]; then
  echo "Missing local web dependencies. Run this once:"
  echo "  npm --prefix $frontend_dir ci"
  exit 1
fi

cleanup() {
  for pid in "${web_pid:-}" "${api_pid:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting the local API…"
(cd "$backend_dir" && exec "$venv_python" -m uvicorn main:app --host 127.0.0.1 --port 8000) &
api_pid=$!

for _ in {1..20}; do
  if ! kill -0 "$api_pid" 2>/dev/null; then
    echo "The local API exited during startup. Check the messages above."
    exit 1
  fi
  if curl --silent --fail http://127.0.0.1:8000/api/health >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl --silent --fail http://127.0.0.1:8000/api/health >/dev/null; then
  echo "The local API did not start. Check the messages above."
  exit 1
fi

echo "Open http://127.0.0.1:5173 (press Ctrl+C here to stop everything)."
(cd "$frontend_dir" && exec node_modules/.bin/vite --host 127.0.0.1 --strictPort) &
web_pid=$!
wait "$web_pid"
