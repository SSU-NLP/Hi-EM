"""pytest configuration — put src/ on sys.path so tests can import hi_em.

A proper pyproject.toml / editable install would remove this, but for
Phase 1 the ``src``-layout with a minimal conftest is simpler and does
not require extra packaging work.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
