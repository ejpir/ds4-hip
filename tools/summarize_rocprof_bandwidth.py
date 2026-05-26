#!/usr/bin/env python3
"""Summarize rocprofv3 counter CSVs as per-kernel memory bandwidth.

rocprofv3 counter CSV schemas vary across ROCm releases, so this parser is
intentionally permissive: it scans each CSV for a header row containing known
kernel/counter columns and accepts either explicit duration columns or
Start/End timestamp pairs.

The important rocprofv3 counters for this repo are:
  FETCH_SIZE  - KiB fetched from memory/cache hierarchy, including overfetch
  WRITE_SIZE  - KiB written

Example:
  python3 tools/summarize_rocprof_bandwidth.py /tmp/ds4_rocprof_bw_*/rocprof
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

KERNEL_KEYS = (
    "Kernel_Name", "Kernel Name", "KernelName", "kernel_name", "Name",
    "Kernel", "kernel", "Symbol", "Function",
)
START_KEYS = ("Start_Timestamp", "Start Timestamp", "BeginNs", "Begin", "Start")
END_KEYS = ("End_Timestamp", "End Timestamp", "EndNs", "End")
DURATION_KEYS = (
    "DurationNs", "Duration_Ns", "Duration(ns)", "Duration (ns)",
    "Duration", "Kernel_Duration", "GPU_Duration",
)
FETCH_KEYS = ("FETCH_SIZE", "FetchSize", "fetch_size")
WRITE_KEYS = ("WRITE_SIZE", "WriteSize", "write_size")
AUX_KEYS = ("L2CacheHit", "MemUnitBusy", "OccupancyPercent", "Wavefronts", "VALUInsts")


def iter_csv_files(paths: Iterable[str]) -> Iterable[Path]:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            yield from sorted(path.rglob("*.csv"))
        elif path.suffix.lower() == ".csv" and path.exists():
            yield path


def norm_float(v: object) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "n/a", "na", "none", "null"}:
        return None
    s = s.replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def pick(row: dict[str, str], keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in row and str(row[k]).strip():
            return row[k]
    lower = {k.lower(): k for k in row}
    for k in keys:
        rk = lower.get(k.lower())
        if rk and str(row[rk]).strip():
            return row[rk]
    return None


def pick_float(row: dict[str, str], keys: Iterable[str]) -> float | None:
    v = pick(row, keys)
    return norm_float(v)


def duration_sec(row: dict[str, str]) -> float | None:
    d = pick_float(row, DURATION_KEYS)
    if d is not None and d > 0.0:
        # rocprof CSV duration fields are normally nanoseconds. If a future
        # schema emits seconds, values will be small; use a conservative guard.
        return d / 1.0e9 if d > 10.0 else d
    st = pick_float(row, START_KEYS)
    en = pick_float(row, END_KEYS)
    if st is None or en is None or en <= st:
        return None
    # rocprof timestamps are nanoseconds.
    return (en - st) / 1.0e9


def find_header(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        cells = [c.strip() for c in next(csv.reader([line]))]
        if not cells:
            continue
        has_counter = any(c in FETCH_KEYS + WRITE_KEYS for c in cells)
        has_kernel = any(c in KERNEL_KEYS for c in cells) or any("kernel" in c.lower() for c in cells)
        if has_counter and has_kernel:
            return i
    return None


def rows_from_csv(path: Path) -> Iterable[dict[str, str]]:
    try:
        text = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    idx = find_header(text)
    if idx is None:
        return
    reader = csv.DictReader(text[idx:])
    for row in reader:
        yield row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="rocprofv3 CSV file(s) or output directory/directories")
    ap.add_argument("--top", type=int, default=30, help="rows to print, default 30")
    ap.add_argument("--min-fetch-mb", type=float, default=0.0, help="hide kernels below this total fetched MB")
    args = ap.parse_args()

    by_kernel: dict[str, dict[str, float]] = {}
    files = list(iter_csv_files(args.paths))
    parsed_rows = 0
    for path in files:
        for row in rows_from_csv(path) or ():
            kernel = pick(row, KERNEL_KEYS)
            if not kernel:
                continue
            sec = duration_sec(row)
            fetch_kib = pick_float(row, FETCH_KEYS) or 0.0
            write_kib = pick_float(row, WRITE_KEYS) or 0.0
            if sec is None or sec <= 0.0:
                continue
            parsed_rows += 1
            d = by_kernel.setdefault(kernel, {"calls": 0.0, "sec": 0.0, "fetch": 0.0, "write": 0.0})
            d["calls"] += 1.0
            d["sec"] += sec
            d["fetch"] += fetch_kib * 1024.0
            d["write"] += write_kib * 1024.0
            for k in AUX_KEYS:
                v = pick_float(row, (k,))
                if v is not None:
                    d[k] = d.get(k, 0.0) + v
                    d[k + "_n"] = d.get(k + "_n", 0.0) + 1.0

    if not by_kernel:
        print(f"no rocprof bandwidth rows found in {len(files)} csv file(s)")
        return 1

    rows = []
    for kernel, d in by_kernel.items():
        fetch_mb = d["fetch"] / 1.0e6
        if fetch_mb < args.min_fetch_mb:
            continue
        sec = d["sec"]
        read_gbs = d["fetch"] / sec / 1.0e9 if sec > 0 else 0.0
        write_gbs = d["write"] / sec / 1.0e9 if sec > 0 else 0.0
        rows.append((d["fetch"], sec, kernel, d, read_gbs, write_gbs))
    rows.sort(reverse=True, key=lambda r: (r[0], r[1]))

    total_fetch = sum(d["fetch"] for _, _, _, d, _, _ in rows)
    total_write = sum(d["write"] for _, _, _, d, _, _ in rows)
    total_sec = sum(d["sec"] for _, _, _, d, _, _ in rows)
    print(f"files={len(files)} rows={parsed_rows} kernels={len(rows)}")
    print(f"sum_kernel_time_ms={total_sec*1e3:.3f} fetch_GB={total_fetch/1e9:.3f} write_GB={total_write/1e9:.3f}")
    if total_sec > 0:
        print(f"aggregate_fetch_GBps={total_fetch/total_sec/1e9:.2f} aggregate_write_GBps={total_write/total_sec/1e9:.2f}")
    print()
    print("calls  time_ms   fetch_GB  write_GB  read_GB/s write_GB/s  kernel")
    for fetch, sec, kernel, d, read_gbs, write_gbs in rows[: args.top]:
        print(f"{int(d['calls']):5d} {sec*1e3:8.3f} {fetch/1e9:9.3f} {d['write']/1e9:9.3f} {read_gbs:9.2f} {write_gbs:10.2f}  {kernel}")
        aux = []
        for k in AUX_KEYS:
            n = d.get(k + "_n", 0.0)
            if n:
                aux.append(f"{k}={d.get(k, 0.0)/n:.2f}")
        if aux:
            print("      " + " ".join(aux))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
