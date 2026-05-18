#!/usr/bin/env python3
"""Compare DS4 router top-k/scores debug dumps.

Expected dump names are produced by DS4_METAL_GRAPH_DUMP_PREFIX, e.g.:
  {prefix}_ffn_moe_topk-20_pos33.i32
  {prefix}_ffn_moe_scores-20_pos33.bin
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np


TOPK_RE = re.compile(r"_ffn_moe_topk-(\d+)_pos(\d+)\.i32$")


def find_topk(prefix):
    out = {}
    for path in glob.glob(prefix + "_ffn_moe_topk-*_pos*.i32"):
        m = TOPK_RE.search(path)
        if not m:
            continue
        key = (int(m.group(2)), int(m.group(1)))  # pos, layer
        out[key] = path
    return out


def score_path(prefix, pos, layer):
    return f"{prefix}_ffn_moe_scores-{layer}_pos{pos}.bin"


def load_i32(path):
    return np.fromfile(path, dtype=np.int32)


def load_f32(path):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.float32)


def margin(scores):
    if scores is None or scores.size < 7:
        return None, None
    order = np.argsort(scores)[::-1]
    return float(scores[order[5]] - scores[order[6]]), order


def fmt(v):
    return "n/a" if v is None else f"{v:.7g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_prefix")
    ap.add_argument("candidate_prefix")
    ap.add_argument("--near", type=float, default=0.01, help="also show rows with 6-vs-7 margin below this")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--all", action="store_true", help="show all common rows")
    args = ap.parse_args()

    base = find_topk(args.baseline_prefix)
    cand = find_topk(args.candidate_prefix)
    keys = sorted(set(base) & set(cand))
    print(f"baseline_rows={len(base)} candidate_rows={len(cand)} common_rows={len(keys)}")

    shown = 0
    changed = 0
    near = 0
    for pos, layer in keys:
        bt = load_i32(base[(pos, layer)])
        ct = load_i32(cand[(pos, layer)])
        if bt.size != ct.size:
            is_changed = True
        else:
            is_changed = not np.array_equal(bt, ct)
        if is_changed:
            changed += 1

        bs = load_f32(score_path(args.baseline_prefix, pos, layer))
        cs = load_f32(score_path(args.candidate_prefix, pos, layer))
        bm, bo = margin(bs)
        cm, co = margin(cs)
        max_delta = None
        if bs is not None and cs is not None and bs.size == cs.size:
            max_delta = float(np.max(np.abs(cs - bs)))
        is_near = (bm is not None and bm <= args.near) or (cm is not None and cm <= args.near)
        if is_near:
            near += 1
        if not args.all and not is_changed and not is_near:
            continue
        if shown >= args.limit:
            continue

        mark = "!" if is_changed else "~" if is_near else "="
        print(
            f"{mark} pos={pos} layer={layer} "
            f"baseline={bt.tolist()} candidate={ct.tolist()} "
            f"margin={fmt(bm)}->{fmt(cm)} max_score_delta={fmt(max_delta)}"
        )
        if is_changed and bs is not None and cs is not None and bo is not None and co is not None:
            union = []
            for x in list(bt) + list(ct) + list(bo[:8]) + list(co[:8]):
                xi = int(x)
                if xi not in union:
                    union.append(xi)
            for e in union[:16]:
                print(f"    e={e:3d} score={float(bs[e]):.7g}->{float(cs[e]):.7g} delta={float(cs[e]-bs[e]):+.7g}")
        shown += 1

    print(f"changed_rows={changed} near_rows={near}")


if __name__ == "__main__":
    main()
