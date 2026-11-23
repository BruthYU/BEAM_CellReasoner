"""Run conda-pack prefix repairs without copying unchanged Docker layer files."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy
import sys


def main():
    prefix = Path(sys.prefix)
    unpack = runpy.run_path(str(prefix / "bin" / "conda-unpack"))
    records = unpack["_prefix_records"]

    def needs_repair(record):
        relative_path, placeholder, mode = record
        data = (prefix / relative_path).read_bytes()
        return unpack["replace_prefix"](data, mode, placeholder, str(prefix)) != data

    # Opening unchanged files in rb+ would copy them from Docker's lower layer.
    # Use the original replacement function to identify changes with read-only I/O.
    with ThreadPoolExecutor(max_workers=8) as executor:
        changed = [
            record for record, needed in zip(records, executor.map(needs_repair, records))
            if needed
        ]
    for relative_path, placeholder, mode in changed:
        unpack["update_prefix"](
            str(prefix / relative_path), str(prefix), placeholder, mode=mode
        )
    print(f"Repaired {len(changed)} of {len(records)} conda-pack prefix records.")


if __name__ == "__main__":
    main()
