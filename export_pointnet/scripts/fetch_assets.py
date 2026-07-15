"""Fetches derived assets for the PointNet export.

Unlike most exports, the pretrained checkpoint is not hosted at a separate
URL -- upstream (yanx27/Pointnet_Pointnet2_pytorch) ships it directly inside
the repo at log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth. So
"fetching" here means copying it out of the submodule checkout (after
`git submodule update --init`) into weights/, verified by SHA256.

Design rules (see openvino-converter skill Section 6):
- Manifest is hardcoded, not scraped at runtime.
- SHA256 mismatch is a hard fail.
- Re-running with everything already present and correct is a no-op.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

EXPORT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LocalAsset:
    src_path: str        # relative to EXPORT_ROOT, inside the PointNet/ submodule
    dest_path: str        # relative to EXPORT_ROOT
    sha256: str
    size_bytes: int
    description: str


ASSETS: list[LocalAsset] = [
    LocalAsset(
        src_path="PointNet/log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth",
        dest_path="weights/pointnet_sem_seg_best.pth",
        sha256="711e15ca18065444707f4351058cabf54b1e8e7f6ee08a209c26f7da24621bbb",
        size_bytes=42_488_257,
        description="PointNet semantic segmentation checkpoint (S3DIS, 13 classes, mIoU 43.7%), shipped by upstream",
    ),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_one(asset: LocalAsset) -> str:
    dest = EXPORT_ROOT / asset.dest_path
    if dest.exists() and sha256_file(dest) == asset.sha256:
        return "skip (already present)"

    src = EXPORT_ROOT / asset.src_path
    if not src.exists():
        raise RuntimeError(
            f"source not found: {src}. Run 'git submodule update --init export_pointnet/PointNet' first."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    shutil.copyfile(src, tmp)
    actual = sha256_file(tmp)
    if actual != asset.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch from {src}: got {actual}, expected {asset.sha256}")
    tmp.replace(dest)
    return f"ok (copied from {asset.src_path})"


def main() -> int:
    failed = 0
    for a in ASSETS:
        print(f"[{a.dest_path}] ({a.size_bytes / 1e6:.1f} MB) {a.description}")
        try:
            status = fetch_one(a)
            print(f"  -> {status}")
        except Exception as e:
            print(f"  -> FAIL: {e}", file=sys.stderr)
            failed += 1

    total = len(ASSETS)
    print(f"\nDone: {total - failed}/{total} assets ready under {EXPORT_ROOT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
