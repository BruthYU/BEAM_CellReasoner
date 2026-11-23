"""CLI import paths and explicit verification of the 2026-11-23 directory migration.

Historical experiment manifests are immutable. A migrated source is accepted only
when both its archived original and its reviewed new file match the layout record.
The ordinary SHA256 helpers continue to return actual file hashes.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
for directory in [SCRIPTS, *[SCRIPTS / n for n in
        ("lib", "data", "training", "evaluation", "checks", "analysis", "launch", "environment")]]:
    if str(directory) not in sys.path:
        sys.path.append(str(directory))


@lru_cache(maxsize=1)
def layout():
    return json.loads((SCRIPTS / "layout.json").read_text())["files"]


def script_path(name):
    """Resolve an existing script basename used by generated source manifests."""
    key = "scripts/" + str(name)
    entry = layout().get(key)
    if entry is not None:
        if entry["new"] is None:
            raise ValueError(f"Retired one-off script: {name}; see scripts/README.md")
        return ROOT / entry["new"]
    path = SCRIPTS / name
    if not path.is_file():
        matches = list(SCRIPTS.glob("*/" + str(name)))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one script named {name}: {matches}")
        path = matches[0]
    return path


@lru_cache(maxsize=1024)
def _digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    path = Path(path)
    st = path.stat()
    return _digest(str(path), st.st_size, st.st_mtime_ns)


def verify_input(name, expected):
    """Validate a current input or the recorded original-to-relocated transition."""
    path = ROOT / name
    if path.is_file() and file_hash(path) == expected:
        return
    try:
        key = str(path.relative_to(ROOT))
    except ValueError:
        key = str(name)
    entry = layout().get(key)
    if entry is None or expected != entry["original_sha256"]:
        raise AssertionError(f"Prepared input changed (not covered by directory migration): {name}")
    if not (ROOT / entry["archive"]).is_file():
        raise FileNotFoundError(f"Historical input verification requires {entry['archive']}; see SETUP.md")
    assert file_hash(ROOT / entry["archive"]) == expected, f"Archived original changed: {name}"
    # Consolidated wrappers depend on newly extracted modules as well.
    for source, digest in json.loads((SCRIPTS / "layout.json").read_text()).get("shared_files", {}).items():
        assert file_hash(ROOT / source) == digest, f"Consolidated dependency changed: {source}"
    if entry["new"] is not None:
        assert file_hash(ROOT / entry["new"]) == entry["new_sha256"], f"Relocated source changed: {entry['new']}"
    else:
        assert entry.get("retired_reason"), f"Missing retirement explanation: {name}"


def relocate_argument(value):
    """Adapt script paths in old frozen AR command plans without editing records."""
    text = str(value)
    for old, entry in layout().items():
        if text == old or text == str(ROOT / old):
            if entry["new"] is None:
                raise ValueError(f"Historical command uses retired script {old}; do not relaunch this one-off job")
            return str(ROOT / entry["new"]) if Path(text).is_absolute() else entry["new"]
    return value
