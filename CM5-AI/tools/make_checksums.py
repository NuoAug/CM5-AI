# -*- coding: utf-8 -*-
"""生成 test-run 的 checksums.sha256（相对路径，排序稳定）。

用法：
  python tools/make_checksums.py test-runs/<run_id>
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
        print("用法：python tools/make_checksums.py <run_dir>")
        return 2
    run = Path(sys.argv[1])
    lines = []
    for path in sorted(p for p in run.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
        rel = path.relative_to(run).as_posix()
        lines.append(f"{sha256(path)}  ./{rel}")
    (run / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{run}: {len(lines)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
