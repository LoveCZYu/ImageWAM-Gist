#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/third_party/LIBERO}"
LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${HOME}/.libero}"

imagewam_run apt-get update
imagewam_run apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
imagewam_run uv pip install mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict thop future cloudpickle opencv-python-headless

if [ ! -d "${LIBERO_DIR}" ]; then
  mkdir -p "$(dirname "${LIBERO_DIR}")"
  imagewam_run git clone "${LIBERO_REPO}" "${LIBERO_DIR}"
fi

touch "${LIBERO_DIR}/libero/libero/__init__.py"
LIBERO_BENCHMARK_INIT="${LIBERO_DIR}/libero/libero/benchmark/__init__.py" imagewam_run imagewam_python - <<'PATCHPY'
import os
from pathlib import Path
path = Path(os.environ['LIBERO_BENCHMARK_INIT'])
text = path.read_text()
old = 'init_states = torch.load(init_states_path)'
new = 'init_states = torch.load(init_states_path, weights_only=False)'
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise RuntimeError(f'Could not patch torch.load in {path}')
PATCHPY

(cd "${LIBERO_DIR}" && imagewam_run uv pip install -e . --force-reinstall)
mkdir -p "${LIBERO_CONFIG_DIR}/config_backups"
cp "${LIBERO_CONFIG_DIR}/config.yaml" "${LIBERO_CONFIG_DIR}/config_backups/config.$(date +%Y%m%d_%H%M%S).yaml" 2>/dev/null || true
rm -f "${LIBERO_CONFIG_DIR}/config.yaml"
