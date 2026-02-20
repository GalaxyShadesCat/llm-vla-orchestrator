#!/usr/bin/env bash
set -euo pipefail

# ==========================
# CONFIGURATION
# ==========================

PYTHON_VERSION="3.11"
OPENPI_URL="https://github.com/Physical-Intelligence/openpi.git"
OPENPI_DIR="openpi"
REQ_FILE="requirements.txt"

need_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "[1/7] Checking prerequisites..."

need_cmd git || { echo "ERROR: git not installed."; exit 1; }

if ! need_cmd uv; then
  echo "[2/7] Installing uv..."
  need_cmd curl || { echo "ERROR: curl not installed."; exit 1; }
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

need_cmd uv || { echo "ERROR: uv not found in PATH."; exit 1; }

echo "[3/7] Creating virtual environment with Python ${PYTHON_VERSION}..."

if [[ ! -d ".venv" ]]; then
  uv venv --python "${PYTHON_VERSION}"
else
  echo "Existing .venv detected, reusing."
fi

echo "[4/7] Initialising/updating openpi submodule..."

if [[ -f ".gitmodules" ]] && git config -f .gitmodules --get submodule.openpi.path >/dev/null 2>&1; then
  git submodule update --init --recursive "${OPENPI_DIR}"
else
  if [[ -d "${OPENPI_DIR}/.git" ]]; then
    (
      cd "${OPENPI_DIR}"
      git pull --ff-only
      git submodule update --init --recursive
    )
  else
    git clone --recurse-submodules "${OPENPI_URL}" "${OPENPI_DIR}"
  fi
fi

echo "[5/7] Installing openpi..."

export GIT_LFS_SKIP_SMUDGE=1
uv pip install --python .venv/bin/python -e "./${OPENPI_DIR}"

echo "[6/7] Installing project requirements..."

if [[ -f "${REQ_FILE}" ]]; then
  uv pip install --python .venv/bin/python -r "${REQ_FILE}"
else
  echo "No ${REQ_FILE} found, skipping."
fi

echo "[7/7] Running sanity check..."

uv run --python .venv/bin/python python - <<'PY'
import sys
print("Python version:", sys.version)
try:
    import openpi
    print("openpi import: OK")
except Exception as e:
    print("openpi import failed:", e)
PY

echo
echo "Setup complete."
echo "Activate environment with:"
echo "  source .venv/bin/activate"
echo
echo "Or run directly:"
echo "  uv run --python .venv/bin/python python your_script.py"