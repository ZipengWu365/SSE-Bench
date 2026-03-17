from __future__ import annotations

import sys
from pathlib import Path


# Ensure repo root is importable when tests are run from other working directories.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

