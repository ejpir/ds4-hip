#!/usr/bin/env python3
"""Compare DS4 KV FP8 quantization decisions from graph dumps.

The tool expects dump files named like:
    {prefix}_KVrope-{layer}_pos{pos}.bin
    {prefix}_KVcur-{layer}_pos{pos}.bin

It recomputes the E4M3FN per-64-value quantization used for the non-RoPE
part of KV rows and reports bucket flips/near-boundary decisions.  It is a
post-processing helper for DS4_METAL_GRAPH_DUMP_PREFIX dumps; it does not need
model weights.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np

FILE_RE = re.compile(r"_KVrope-(\d+)_pos(\d+)\.bin$")


def e4m3_value(i: int) -> float:
    exp_scale = [
        0.0, 0.015625, 0.03125, 0.0625,
        0.125, 0.25, 0.5, 1.0,
        2.0, 4.0, 8.0, 16.0,
        32.0, 64.0, 128.0, 256.0,
    ]
    exp = (i >> 3) & 0x0F
    mant = i & 0x07
    if exp == 0:
        return mant * 0.001953125
    return (1.0 + mant * 0.125) * exp_scale[exp]


E4M3 = np.array([e4m3_value(i) for i in range(127)], dtype=np.float64)


def quantize_abs_bucket(qabs: float) -> int:
    qabs = min(abs(float(qabs)), 448.0)
    best = int(np.searchsorted(E4M3, qabs, side="right") - 1)
    best = max(0, min(126, best))
    if best < 126:
        best_diff = abs(qabs - E4M3[best])
        next_diff = abs(qabs - E4M3[best + 1])
        if next_diff < best_diff or (
            next_diff == best_diff and ((best + 1) & 1) == 0 and (best & 1) != 0
        ):
            best += 1
    return best


def scale_for_block(x: np.ndarray) -> float:
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if amax < 1.0e-4:
        amax = 1.0e-4
    return math.ldexp(1.0, math.ceil(math.log2(amax / 448.0)))


def nearest_boundary(bucket: int, qabs: float) -> tuple[float, float]:
    """Return (boundary, signed distance qabs-boundary) for nearest boundary."""
    candidates: list[float] = []
    if bucket > 0:
        candidates.append(0.5 * (E4M3[bucket - 1] + E4M3[bucket]))
    if bucket < 126:
        candidates.append(0.5 * (E4M3[bucket] + E4M3[bucket + 1]))
    if not candidates:
        return 0.0, float("inf")
    boundary = min(candidates, key=lambda b: abs(qabs - b))
    return boundary, qabs - boundary


@dataclass
class Decision:
    dim: int
    block: int
    value: float
    cur: float
    scale: float
    q: float
    bucket: int
    boundary: float
    boundary_dist: float


def decisions(rope: np.ndarray, cur: np.ndarray, head_dim: int, n_rot: int) -> list[Decision]:
    n_nope = head_dim - n_rot
    out: list[Decision] = []
    for off in range(0, n_nope, 64):
        block = rope[off:off + 64]
        scale = scale_for_block(block)
        for i in range(block.size):
            dim = off + i
            q = float(rope[dim]) / scale
            bucket = quantize_abs_bucket(q)
            boundary, dist = nearest_boundary(bucket, abs(q))
            out.append(Decision(
                dim=dim,
                block=off // 64,
                value=float(rope[dim]),
                cur=float(cur[dim]),
                scale=scale,
                q=q,
                bucket=bucket,
                boundary=boundary,
                boundary_dist=dist,
            ))
    return out


def parse_u32_filter(spec: str | None) -> set[int] | None:
    if not spec or spec == "all":
        return None
    vals: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo_s, hi_s = part.split('-', 1)
            lo, hi = int(lo_s), int(hi_s)
            vals.update(range(lo, hi + 1))
        else:
            vals.add(int(part))
    return vals


def discover(prefix: str) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for path in glob.glob(prefix + "_KVrope-*_pos*.bin"):
        m = FILE_RE.search(path)
        if not m:
            continue
        out[(int(m.group(1)), int(m.group(2)))] = path
    return out


def read_f32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def fmt_dec(d: Decision) -> str:
    return (
        f"dim={d.dim:3d} blk={d.block} val={d.value:+.9g} cur={d.cur:+.9g} "
        f"scale={d.scale:.9g} q={d.q:+.7g} bucket={d.bucket:3d} "
        f"bdry={d.boundary:.7g} dist={d.boundary_dist:+.3g}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_prefix")
    ap.add_argument("candidate_prefix")
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--n-rot", type=int, default=64)
    ap.add_argument("--layer", default=None, help="layer filter, e.g. 5 or 0-5 or all")
    ap.add_argument("--pos", default=None, help="position filter, e.g. 24 or 24,29 or all")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--near", type=float, default=0.02, help="report unchanged decisions within this q-space distance of a boundary")
    args = ap.parse_args()

    layers = parse_u32_filter(args.layer)
    poss = parse_u32_filter(args.pos)
    base = discover(args.baseline_prefix)
    cand = discover(args.candidate_prefix)
    keys = sorted(set(base) & set(cand))
    if layers is not None:
        keys = [k for k in keys if k[0] in layers]
    if poss is not None:
        keys = [k for k in keys if k[1] in poss]

    print(f"baseline_rows={len(base)} candidate_rows={len(cand)} common_rows={len(keys)}")
    total_flips = 0
    total_near = 0
    shown = 0
    for layer, pos in keys:
        b_rope = read_f32(base[(layer, pos)])
        c_rope = read_f32(cand[(layer, pos)])
        b_cur_path = f"{args.baseline_prefix}_KVcur-{layer}_pos{pos}.bin"
        c_cur_path = f"{args.candidate_prefix}_KVcur-{layer}_pos{pos}.bin"
        if not os.path.exists(b_cur_path) or not os.path.exists(c_cur_path):
            continue
        b_cur = read_f32(b_cur_path)
        c_cur = read_f32(c_cur_path)
        if b_rope.size < args.head_dim or c_rope.size < args.head_dim:
            continue
        b_dec = decisions(b_rope, b_cur, args.head_dim, args.n_rot)
        c_dec = decisions(c_rope, c_cur, args.head_dim, args.n_rot)
        flips = []
        near = []
        for bd, cd in zip(b_dec, c_dec):
            cur_delta = cd.cur - bd.cur
            bucket_flip = bd.bucket != cd.bucket or math.copysign(1.0, bd.cur or 1.0) != math.copysign(1.0, cd.cur or 1.0)
            if bucket_flip or abs(cur_delta) > 1.0e-7:
                flips.append((abs(cur_delta), cur_delta, bd, cd))
            elif min(abs(bd.boundary_dist), abs(cd.boundary_dist)) <= args.near:
                near.append((min(abs(bd.boundary_dist), abs(cd.boundary_dist)), bd, cd))
        total_flips += len(flips)
        total_near += len(near)
        if flips or near:
            max_cur = max([x[0] for x in flips], default=0.0)
            min_near = min([x[0] for x in near], default=float("inf"))
            print(f"\nlayer={layer} pos={pos} flips={len(flips)} near={len(near)} max_cur_delta={max_cur:.9g} min_near={min_near:.3g}")
            for abs_delta, cur_delta, bd, cd in sorted(flips, key=lambda x: x[0], reverse=True)[:args.limit]:
                print(f"  FLIP cur_delta={cur_delta:+.9g}")
                print(f"    base {fmt_dec(bd)}")
                print(f"    cand {fmt_dec(cd)}")
                shown += 1
            remaining = max(0, args.limit - len(flips))
            for _, bd, cd in sorted(near, key=lambda x: x[0])[:remaining]:
                print("  NEAR")
                print(f"    base {fmt_dec(bd)}")
                print(f"    cand {fmt_dec(cd)}")
                shown += 1
    print(f"\nchanged_decisions={total_flips} near_decisions={total_near} shown={shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
