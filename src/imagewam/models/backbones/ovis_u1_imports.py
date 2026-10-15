from __future__ import annotations

import sys
from pathlib import Path


def ensure_ovis_u1_remote_code_importable() -> None:
    """Expose vendored Ovis-U1 HF remote code as `ovis_u1_hf`."""
    repo_root = Path(__file__).resolve().parents[4]
    third_party = repo_root / "third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
