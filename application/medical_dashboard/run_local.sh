#!/usr/bin/env bash
# Build once with --setup, then serve the app and API on a single loopback port.
set -euo pipefail

dashboard_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$dashboard_dir/../.." && pwd)"
venv_python="$repo_dir/.venv/bin/python"
frontend_dir="$dashboard_dir/frontend"
backend_dir="$dashboard_dir/backend"
port=5173
setup=false

usage() {
  cat <<'HELP'
Usage: bash application/medical_dashboard/run_local.sh [--setup]
       [--gallery FILE] [--embeddings FILE | --checkpoint FILE] [--port PORT]

Use --setup on the first run to install dependencies and build the dashboard.
Pass a gallery CSV and a keyed embedding archive, or a UniMedCLIP checkpoint
to compute embeddings. The equivalent MEDCLIPSEG_* environment variables work too.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup) setup=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --gallery|--embeddings|--checkpoint|--port)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Missing value for $1." >&2
        exit 2
      fi
      case "$1" in
        --gallery) export MEDCLIPSEG_GALLERY="$2" ;;
        --embeddings) export MEDCLIPSEG_EMBEDDINGS="$2" ;;
        --checkpoint)
          export MEDCLIPSEG_CHECKPOINT="$2"
          unset MEDCLIPSEG_EMBEDDINGS ;;
        --port) port="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$setup" == true ]]; then
  if [[ ! -x "$venv_python" ]]; then
    python3 -m venv "$repo_dir/.venv"
  fi
  requirements="$backend_dir/requirements.txt"
  if [[ -z "${MEDCLIPSEG_EMBEDDINGS:-}" ]]; then
    requirements="$backend_dir/requirements-encoder.txt"
  fi
  "$venv_python" -m pip install -r "$requirements"
  npm --prefix "$frontend_dir" ci --include=dev
  npm --prefix "$frontend_dir" run build
fi

if [[ ! -x "$venv_python" || ! -f "$frontend_dir/dist/index.html" ]] ||
   ! "$venv_python" -c 'import fastapi, uvicorn, multipart, faiss, PIL, pandas, matplotlib' 2>/dev/null; then
  echo "The dashboard has not been installed. Rerun with --setup." >&2
  exit 1
fi

echo "Open http://127.0.0.1:$port to use MedCLIPSeg++."
exec "$venv_python" -m uvicorn main:app --app-dir "$backend_dir" --host 127.0.0.1 --port "$port"
