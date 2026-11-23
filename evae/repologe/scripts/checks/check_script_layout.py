"""Check a code checkout; --historical additionally requires local experiment archives."""
# python3 scripts/checks/check_script_layout.py
# python3 scripts/checks/check_script_layout.py --historical  # requires external archives

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true", help="Also verify external historical snapshots and experiment metadata")
    args = parser.parse_args()
    root = _bootstrap.ROOT
    archive = root / "artifacts/repository_cleanup/20261123_scripts_layout"
    metadata = {}
    if args.historical:
        required = [root / info["archive"] for info in _bootstrap.layout().values()]
        required.append(archive / "experiment_metadata_before.json")
        missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
        if missing:
            parser.error("Historical audit requires external archives; see SETUP.md. Missing: " + ", ".join(missing[:5]))
    files = _bootstrap.layout()
    moved = retired = 0
    for old, info in files.items():
        if args.historical:
            assert _bootstrap.file_hash(root / info["archive"]) == info["original_sha256"]
        if info["new"] is None:
            assert not (root / old).exists()
            retired += 1
        else:
            assert _bootstrap.file_hash(root / info["new"]) == info["new_sha256"], info["new"]
            if info["new"] != old:
                assert not (root / old).exists()
                moved += 1
        if args.historical:
            _bootstrap.verify_input(old, info["original_sha256"])
    for name, digest in json.loads((root / "scripts/layout.json").read_text()).get("shared_files", {}).items():
        assert _bootstrap.file_hash(root / name) == digest, f"Shared source changed: {name}"
    # The portable mode also rejects incorrect hashes; it never grants a bypass.
    try:
        _bootstrap.verify_input("scripts/" + "replogle_control_baseline.py", "0" * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect original hash accepted")
    if args.historical:
        metadata = json.loads((archive / "experiment_metadata_before.json").read_text())
        for name, before in metadata.items():
            assert _bootstrap.file_hash(root / name) == before, f"Historical metadata rewritten: {name}"
    current = list((root / "scripts").rglob("*.py"))
    script_names = {Path(k).stem for k in files if k.endswith(".py")}
    modules = {p.stem: p for p in current}
    assert len(modules) == len(current), "Ambiguous flat script imports"
    for p in current:
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in script_names:
                assert node.module in modules, f"{p}: imports retired module {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in script_names:
                        assert alias.name in modules, f"{p}: imports retired module {alias.name}"
    shells = list((root / "scripts").rglob("*.sh"))
    for p in shells:
        subprocess.run(["bash", "-n", str(p)], check=True)
    # All maintained literal script paths must resolve (original root entry points included).
    for p in [*current, *shells, root / "Dockerfile"]:
        for name in re.findall(r"scripts/(?:[a-z_]+/)*[a-zA-Z0-9_]+\.(?:py|sh)", p.read_text()):
            assert (root / name).is_file(), f"{p}: broken script path {name}"
    docs = [root / "README.md", root / "scripts/README.md", root / "SETUP.md", *list((root / "docs").glob("*.md"))]
    checked_links = 0
    for p in docs:
        for url in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            # Only local script/doc links changed by this task; historical external URLs stay as-is.
            if "://" in url or url.startswith("#"):
                continue
            url = url.split("#")[0]
            if "scripts/" in url or "docs/" in url or url.endswith((".md", ".html")):
                assert (p.parent / url).exists(), f"{p}: broken link {url}"
                checked_links += 1
    print(json.dumps(dict(status="passed", moved_scripts=moved, retired_scripts=retired,
        python_syntax=len(current), shell_syntax=len(shells), checked_links=checked_links,
        historical_audit="passed" if args.historical else "not_requested",
        historical_metadata_unchanged=len(metadata) if args.historical else None, jobs_submitted=0), indent=2))


if __name__ == "__main__":
    main()
