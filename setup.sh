#!/usr/bin/env bash
set -euo pipefail

# ==========================
# CONFIGURATION
# ==========================

PYTHON_VERSION="3.11"
OPENPI_URL="https://github.com/Physical-Intelligence/openpi.git"
OPENPI_DIR="openpi"
REQ_FILE="requirements.txt"
VENV_DIR=".venv"

need_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "[1/7] Checking prerequisites..."

need_cmd git || { echo "ERROR: git not installed."; exit 1; }
need_cmd curl || { echo "ERROR: curl not installed."; exit 1; }

if ! need_cmd uv; then
  echo "[2/7] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

need_cmd uv || { echo "ERROR: uv not found in PATH."; exit 1; }

echo "[3/7] Creating virtual environment (Python ${PYTHON_VERSION})..."

if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
  echo "Existing ${VENV_DIR} detected, reusing."
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[4/7] Upgrading pip..."

"${PYTHON_BIN}" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${PYTHON_BIN}" -m pip install --upgrade pip

echo "[5/7] Initialising/updating openpi..."

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

echo "[6/7] Installing openpi (pinned, non-editable)..."

export GIT_LFS_SKIP_SMUDGE=1
"${PYTHON_BIN}" -m pip install "./${OPENPI_DIR}"

echo "[7/7] Installing project requirements..."

if [[ -f "${REQ_FILE}" ]]; then
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
else
  echo "No ${REQ_FILE} found, skipping."
fi

echo
echo "Running sanity check..."

"${PYTHON_BIN}" - <<'PY'
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
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Or run directly:"
echo "  ${VENV_DIR}/bin/python your_script.py"