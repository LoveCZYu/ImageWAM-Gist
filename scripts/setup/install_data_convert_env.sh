#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

DATA_CONVERT_VENV="${DATA_CONVERT_VENV:-${REPO_ROOT}/.venv-data-convert}"
UV_INDEX_URL="${UV_INDEX_URL:-}"

imagewam_print_config DATA_CONVERT_VENV UV_INDEX_URL
imagewam_run uv venv --python "${DATA_CONVERT_PYTHON:-3.11}" "${DATA_CONVERT_VENV}"
# shellcheck disable=SC1090
source "${DATA_CONVERT_VENV}/bin/activate"
imagewam_run uv pip install mcap
if [ -n "${UV_INDEX_URL}" ]; then
  imagewam_run uv pip install tairos-data-convert --index-url "${UV_INDEX_URL}" --extra-index-url https://pypi.org/simple --upgrade
else
  imagewam_run uv pip install tairos-data-convert --upgrade
fi
