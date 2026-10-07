"""LayoutGKN adapted to our task, imported from the pinned upstream checkout.

Upstream is not copied: it lives at commit 395dc92 in `external/LayoutGKN`
(see `external/LayoutGKN.SOURCE`). `upstream()` adds it to `sys.path`; call it before importing `LayoutGKN.*`.
"""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM_SRC = Path("/work/cvcs2026/ai_interior_design/external/LayoutGKN/src")
UPSTREAM_CONF = UPSTREAM_SRC.parent / "conf" / "default.yaml"
UPSTREAM_COMMIT = "395dc92"


def upstream() -> None:
    """Make the pinned upstream package importable as `LayoutGKN`."""
    if not UPSTREAM_SRC.is_dir():
        raise FileNotFoundError(f"upstream LayoutGKN missing: {UPSTREAM_SRC}")
    p = str(UPSTREAM_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
