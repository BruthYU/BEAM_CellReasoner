"""Resolve historical Stage1 records without legacy filesystem aliases."""
from __future__ import annotations

from pathlib import Path


STAGE1_RELOCATIONS = {
    "celltok_stage1_shared": ("stage1", "full_primary"),
    "celltok_stage1_shared_nonprimary": ("stage1", "nonprimary"),
    "celltok_stage1": ("stage1", "legacy"),
}


def resolve_data_path(value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Translate only the three known data roots; leave frozen records intact.

    Relative archive entries remain relative to their manifest's directory.
    Translation also handles absolute paths embedded in older run metadata.
    """
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    parts = path.parts
    for i in range(len(parts) - 1):
        if parts[i] == "data" and parts[i + 1] in STAGE1_RELOCATIONS:
            path = Path(*parts[:i + 1], *STAGE1_RELOCATIONS[parts[i + 1]], *parts[i + 2:])
            break
    return path.resolve()
