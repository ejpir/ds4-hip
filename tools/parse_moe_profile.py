#!/usr/bin/env python3
"""Summarize DS4 HIP MoE routing dumps and HIP profile logs.

Examples:
  python3 tools/parse_moe_profile.py --routing /tmp/ds4-routing2.log
  python3 tools/parse_moe_profile.py --profile /tmp/moe_profile_rpb16_rpb16.err
  python3 tools/parse_moe_profile.py --routing /tmp/ds4-routing2.log --first 43 \
      --profile /tmp/moe_profile_rpb16_rpb16.err
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path
from typing import Iterable

CurveWhatIf = tuple[str, float, list[tuple[int, float]]]


def percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = math.ceil(len(sorted_vals) * pct) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def parse_routing(path: Path) -> list[dict]:
    dumps: list[dict] = []
    cur: dict | None = None
    dump_re = re.compile(r"HIP MoE routing dump #(\d+) tokens=(\d+).*assignments=(\d+)")
    counts_re = re.compile(r"e\d+:\s+(.*)$")

    for line in path.read_text(errors="ignore").splitlines():
        m = dump_re.search(line)
        if m:
            if cur is not None and len(cur["counts"]) == 256:
                dumps.append(cur)
            cur = {
                "idx": int(m.group(1)),
                "tokens": int(m.group(2)),
                "assignments": int(m.group(3)),
                "counts": [],
                "summary": line,
            }
            continue
        if cur is None:
            continue
        m = counts_re.search(line)
        if not m:
            continue
        cur["counts"].extend(int(x) for x in m.group(1).split())
        if len(cur["counts"]) == 256:
            dumps.append(cur)
            cur = None

    if cur is not None and len(cur["counts"]) == 256:
        dumps.append(cur)
    return dumps


def parse_what_if(specs: list[str]) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise SystemExit(f"bad --what-if item {part!r}; expected THRESHOLD:SPEEDUP")
            t_s, s_s = part.split(":", 1)
            out.append((int(t_s), float(s_s)))
    return out


def parse_curve_what_if(specs: list[str]) -> list[CurveWhatIf]:
    """Parse NAME:BASE_MS:THRESHOLD:SPEEDUP[,THRESHOLD:SPEEDUP...]."""
    out: list[CurveWhatIf] = []
    for spec in specs:
        fields = spec.split(":", 2)
        if len(fields) != 3:
            raise SystemExit(
                f"bad --curve-what-if {spec!r}; expected NAME:BASE_MS:THRESHOLD:SPEEDUP[,THRESHOLD:SPEEDUP...]"
            )
        name, base_ms_s, curve_s = fields
        curve = parse_what_if([curve_s])
        if not name or not curve:
            raise SystemExit(f"bad --curve-what-if {spec!r}; empty name or curve")
        out.append((name, float(base_ms_s), sorted(curve)))
    return out


def summarize_routing(
    dumps: list[dict],
    title: str,
    thresholds: Iterable[int],
    what_ifs: list[tuple[int, float]],
    curve_what_ifs: list[CurveWhatIf],
) -> None:
    print(f"== routing: {title} ==")
    if not dumps:
        print("no complete routing count dumps found\n")
        return

    total_assign = 0
    layers = []
    work = {t: 0 for t in thresholds}
    expert_hits = {t: 0 for t in thresholds}
    top_work = {7: 0, 22: 0, 54: 0}

    for d in dumps:
        counts = d["counts"]
        total = sum(counts)
        total_assign += total
        nz = sorted(c for c in counts if c > 0)
        sorted_counts = sorted(counts, reverse=True)
        layers.append({
            "active": sum(c > 0 for c in counts),
            "empty": sum(c == 0 for c in counts),
            "max": max(counts),
            "p50": percentile(nz, 0.50),
            "p75": percentile(nz, 0.75),
            "p90": percentile(nz, 0.90),
            "p95": percentile(nz, 0.95),
            "p99": percentile(nz, 0.99),
            "total": total,
        })
        for t in thresholds:
            expert_hits[t] += sum(c >= t for c in counts)
            work[t] += sum(c for c in counts if c >= t)
        for k in top_work:
            top_work[k] += sum(sorted_counts[:k])

    tokens = sorted(set(d["tokens"] for d in dumps))
    print(f"dumps={len(dumps)} tokens={tokens} assignments={total_assign}")
    for key in ("active", "empty", "max", "p50", "p75", "p90", "p95", "p99"):
        vals = [x[key] for x in layers]
        print(
            f"{key:>6}: avg={statistics.mean(vals):7.2f} "
            f"p50={statistics.median(vals):7.2f} min={min(vals):5d} max={max(vals):5d}"
        )
    print("threshold  experts/layer  work_share")
    for t in thresholds:
        print(f">={t:<4d} {expert_hits[t] / len(dumps):13.2f} {100.0 * work[t] / max(1, total_assign):9.2f}%")
    for k, v in top_work.items():
        print(f"top{k:<2d} work_share={100.0 * v / max(1, total_assign):.2f}%")
    if what_ifs:
        print("what-if assignment-weighted speedup")
        for threshold, speedup in what_ifs:
            hot = sum(sum(c for c in d["counts"] if c >= threshold) for d in dumps)
            hot_share = hot / max(1, total_assign)
            total_speedup = 1.0 / ((hot_share / speedup) + (1.0 - hot_share))
            print(
                f"hot>={threshold:<4d} hot_share={100.0 * hot_share:6.2f}% "
                f"hot_speedup={speedup:5.2f}x ideal_total={total_speedup:5.3f}x"
            )
    if curve_what_ifs:
        print("curve what-if assignment-weighted stage time")
        all_counts = [c for d in dumps for c in d["counts"]]
        for name, base_ms, curve in curve_what_ifs:
            def speed_for(c: int) -> float:
                s = 1.0
                for threshold, speedup in curve:
                    if c >= threshold:
                        s = speedup
                    else:
                        break
                return s
            new_frac = sum(c / speed_for(c) for c in all_counts) / max(1, total_assign)
            new_ms = base_ms * new_frac
            curve_desc = ",".join(f">={t}:{s:g}x" for t, s in curve)
            print(
                f"{name:16s} base={base_ms:8.2f} ms new={new_ms:8.2f} ms "
                f"save={base_ms - new_ms:8.2f} ms frac={new_frac:6.3f} curve={curve_desc}"
            )
    print()


def parse_profile(path: Path) -> dict[str, list[float]]:
    prof_re = re.compile(r"HIP profile (\S+) in=(\d+) out=(\d+) tokens=(\d+) ([0-9.]+) ms")
    out: dict[str, list[float]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = prof_re.search(line)
        if not m:
            continue
        out.setdefault(m.group(1), []).append(float(m.group(5)))
    return out


def summarize_profile(path: Path) -> None:
    prof = parse_profile(path)
    print(f"== profile: {path} ==")
    if not prof:
        print("no HIP profile lines found\n")
        return
    total = 0.0
    for label in sorted(prof):
        vals = prof[label]
        s = sum(vals)
        total += s
        vals_sorted = sorted(vals)
        print(
            f"{label:32s} n={len(vals):4d} total={s:10.2f} ms "
            f"avg={statistics.mean(vals):8.2f} p50={statistics.median(vals):8.2f} "
            f"p90={vals_sorted[math.ceil(len(vals_sorted) * 0.90) - 1]:8.2f} "
            f"min={min(vals):8.2f} max={max(vals):8.2f}"
        )
    print(f"profiled_total_ms={total:.2f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routing", action="append", default=[], help="stderr/log file containing DS4_HIP_MOE_ROUTING_DUMP_COUNTS output")
    ap.add_argument("--profile", action="append", default=[], help="stderr/log file containing ds4 HIP profile lines")
    ap.add_argument("--first", type=int, default=0, help="only summarize the first N complete routing dumps")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N complete routing dumps before summarizing")
    ap.add_argument("--thresholds", default="1,8,16,32,64,128,256,512", help="comma-separated bucket thresholds")
    ap.add_argument("--what-if", action="append", default=[], help="assignment-weighted speedup estimate, e.g. 64:1.3,128:2.0")
    ap.add_argument(
        "--curve-what-if",
        action="append",
        default=[],
        help="stage what-if using bucket-size speed curve: NAME:BASE_MS:THRESHOLD:SPEEDUP[,THRESHOLD:SPEEDUP...]",
    )
    args = ap.parse_args()

    thresholds = [int(x) for x in args.thresholds.split(",") if x]
    what_ifs = parse_what_if(args.what_if)
    curve_what_ifs = parse_curve_what_if(args.curve_what_if)
    for name in args.routing:
        dumps = parse_routing(Path(name))
        if args.skip:
            dumps = dumps[args.skip:]
        if args.first:
            dumps = dumps[:args.first]
        summarize_routing(dumps, name, thresholds, what_ifs, curve_what_ifs)
    for name in args.profile:
        summarize_profile(Path(name))
    if not args.routing and not args.profile:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
