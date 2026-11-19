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

if ! "$venv_python" -c 'import fastapi, uvicorn, multipart, numpy, PIL, torch, faiss' 2>/dev/null; then
  echo "Missing local Python dependencies. Run this once while connected to your package mirror:"
  echo "  $venv_python -m pip install -r $backend_dir/requirements.txt"
  exit 1
fi

if [[ ! -d "$frontend_dir/node_modules" ]]; then
  echo "Missing local web dependencies. Run this once:"
  echo "  npm --prefix $frontend_dir install"
  exit 1
fi

cleanup() {
  [[ -n "${api_pid:-}" ]] && kill "$api_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting the local API…"
(cd "$backend_dir" && exec "$venv_python" -m uvicorn main:app --host 127.0.0.1 --port 8000) &
api_pid=$!

for _ in {1..20}; do
  if curl --silent --fail http://127.0.0.1:8000/api/health >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl --silent --fail http://127.0.0.1:8000/api/health >/dev/null; then
  echo "The local API did not start. Check the messages above."
  exit 1
fi

echo "Opening http://127.0.0.1:5173 (press Ctrl+C here to stop everything)."
open http://127.0.0.1:5173
cd "$frontend_dir"
exec npm run dev -- --host 127.0.0.1
