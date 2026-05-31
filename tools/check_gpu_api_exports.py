#!/usr/bin/env python3
"""Check that a backend object/binary exports every ds4_gpu_* API in ds4_gpu.h.

This is intentionally lightweight: it does not build anything, it only parses
header declarations and compares them with `nm` output from an already-built
object or executable.

Example:
  python3 tools/check_gpu_api_exports.py --backend ds4_rocm.o
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

API_DECL_RE = re.compile(r"^\s*[A-Za-z_][\w\s]*\*?\s*(ds4_gpu_[A-Za-z0-9_]+)\s*\(")
EXPORTED_TYPES = {"T", "D", "B", "R", "W"}


def parse_header(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = API_DECL_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if name in seen:
            raise SystemExit(f"duplicate declaration in {path}:{line_no}: {name}")
        seen.add(name)
        names.append(name)
    if not names:
        raise SystemExit(f"no ds4_gpu_* declarations found in {path}")
    return names


def nm_exports(path: Path, nm: str) -> set[str]:
    try:
        proc = subprocess.run(
            [nm, "-g", "--defined-only", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"nm executable not found: {nm}") from exc
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"{nm} failed for {path}")

    exports: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[-1]
        sym_type = parts[-2] if len(parts) >= 3 else parts[0]
        if name.startswith("ds4_gpu_") and sym_type in EXPORTED_TYPES:
            exports.add(name)
    return exports


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", default="ds4_gpu.h", type=Path,
                        help="API header to parse (default: ds4_gpu.h)")
    parser.add_argument("--backend", default="ds4_rocm.o", type=Path,
                        help="backend object/binary to inspect (default: ds4_rocm.o)")
    parser.add_argument("--nm", default="nm", help="nm executable (default: nm)")
    parser.add_argument("--warn-extra", action="store_true",
                        help="also print backend ds4_gpu_* exports not declared in the header")
    args = parser.parse_args(argv)

    if not args.header.is_file():
        raise SystemExit(f"header not found: {args.header}")
    if not args.backend.is_file():
        raise SystemExit(f"backend object/binary not found: {args.backend}")

    declared = parse_header(args.header)
    exported = nm_exports(args.backend, args.nm)
    missing = [name for name in declared if name not in exported]
    extra = sorted(name for name in exported if name not in set(declared))

    if missing:
        print(f"missing {len(missing)} ds4_gpu_* export(s) in {args.backend}:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(f"ok: {args.backend} exports all {len(declared)} ds4_gpu_* APIs declared in {args.header}")
    if args.warn_extra and extra:
        print(f"warning: {args.backend} has {len(extra)} extra ds4_gpu_* export(s):", file=sys.stderr)
        for name in extra:
            print(f"  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
