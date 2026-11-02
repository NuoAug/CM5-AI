# -*- coding: utf-8 -*-
"""校验 test-run 目录完整性：对比 checksums.sha256 与实际文件 SHA256。

用法：
  python tools/verify_run.py test-runs/<run_id>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("用法：python tools/verify_run.py <run_dir>")
        return 2
    run = Path(sys.argv[1])
    sums = run / "checksums.sha256"
    if not sums.exists():
        print("缺少 checksums.sha256；先在源端生成：find . -type f ! -name checksums.sha256 -exec sha256sum {} + > checksums.sha256")
        return 2
    missing, mismatch, checked = [], [], 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        path = run / rel
        if not path.exists():
            missing.append(rel)
            continue
        if sha256(path) != digest:
            mismatch.append(rel)
        else:
            checked += 1
    print(f"checked={checked} mismatch={len(mismatch)} missing={len(missing)}")
    for item in mismatch:
        print("MISMATCH", item)
    for item in missing:
        print("MISSING", item)
    return 0 if not mismatch and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
