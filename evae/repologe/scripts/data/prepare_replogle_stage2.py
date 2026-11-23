"""Fetch the original config's Arc ESM2 feature file and audit Replogle coverage.

Only the selected ZIP member is downloaded using HTTP ranges. No expression
data or training pipeline is changed.
"""

# Support direct CLI execution and shared imports after directory organization.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import _bootstrap


import hashlib
import io
import json
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import h5py
import torch

from prepare_replogle_stage1 import ROOT, read_column

SOURCE = "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip"
GENERATION = "1763329752965344"
PROXY = "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
MEMBER = "competition_support_set/ESM2_pert_features.pt"


def request(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        return urllib.request.urlopen(req, timeout=60)
    except (urllib.error.URLError, TimeoutError):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
        return opener.open(req, timeout=60)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("Negative ZIP offset")
        return self.pos

    def read(self, size=-1):
        size = min(self.size - self.pos, size if size >= 0 else self.size - self.pos)
        if size <= 0:
            return b""
        start = self.pos
        with request(self.url, {"Range": f"bytes={start}-{start + size - 1}"}) as response:
            assert response.status == 206, "Server must honor byte ranges"
            assert response.headers["Content-Range"].startswith(f"bytes {start}-")
            data = response.read()
        assert len(data) == size
        self.pos += size
        return data


def main():
    artifacts = ROOT / "artifacts/replogle_stage2"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "datasets" / MEMBER
    provenance_path = artifacts / "embedding_source.json"
    if not destination.exists():
        pinned = SOURCE + "?generation=" + GENERATION
        with request(pinned, {"Range": "bytes=0-0"}) as response:
            assert response.status == 206
            size = int(response.headers["Content-Range"].split("/")[-1])
            generation = response.headers.get("x-goog-generation")
            assert generation == GENERATION
            response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.part")
        with zipfile.ZipFile(RemoteZip(pinned, size)) as archive:
            info = archive.getinfo(MEMBER)
            total, progress = 0, 64 * 1024**2
            with archive.open(info) as source, temporary.open("wb") as output:
                while block := source.read(8 * 1024**2):
                    output.write(block)
                    total += len(block)
                    if total >= progress:
                        print(f"ESM2: extracted {total / 1024**2:.0f}/{info.file_size / 1024**2:.0f} MiB", flush=True)
                        progress += 64 * 1024**2
            assert total == info.file_size
            # ZipExtFile verifies the member CRC at EOF.
            provenance = {"source_url": pinned, "generation": generation,
                          "member": MEMBER, "member_crc32": f"{info.CRC:08x}",
                          "member_size": info.file_size, "compressed_member_size": info.compress_size,
                          "zip_crc_verified": True,
                          "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest()}
        temporary.replace(destination)
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    else:
        provenance = json.loads(provenance_path.read_text())
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == provenance["sha256"]

    embeddings = torch.load(destination, map_location="cpu", weights_only=True)
    assert isinstance(embeddings, dict) and embeddings
    dimensions = {tuple(v.shape) for v in embeddings.values()}
    assert len(dimensions) == 1, dimensions
    shape = next(iter(dimensions))
    assert len(shape) == 1
    assert all(torch.isfinite(v).all() for v in embeddings.values())
    labels = set()
    prep = json.loads((ROOT / "artifacts/replogle_stage1/data_preparation.json").read_text())
    for file in prep["files_in_loader_order"]:
        with h5py.File(file["source"], "r") as data:
            labels.update(read_column(data["obs"], "gene").tolist())
    heldout = json.loads((ROOT / "assets/scldm_replogle_test.json").read_text())["pert"]
    missing = sorted(labels - set(embeddings))
    zero = sorted(g for g in labels if g in embeddings and not embeddings[g].count_nonzero())
    report = {"status": "passed", "path": str(destination), "source": provenance,
              "feature_dim": shape[0], "embedding_keys": len(embeddings),
              "dataset_labels": len(labels), "matched_labels": len(labels - set(missing)),
              "missing_labels": missing, "present_but_zero_labels": zero,
              "heldout_targets": len(heldout),
              "heldout_missing_or_zero": sorted(set(heldout) & (set(missing) | set(zero))),
              "missing_feature_behavior": "Original loader uses zero vectors for missing labels",
              "required_labels": sorted(labels)}
    (artifacts / "raw_embedding_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "required_labels"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
