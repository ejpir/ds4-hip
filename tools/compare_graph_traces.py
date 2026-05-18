#!/usr/bin/env python3
"""Compare DS4_METAL_GRAPH_TRACE_FILE JSONL traces.

The trace is intentionally compact: it records tensor hashes and per-tensor
summary stats at each debug hook.  This tool lines up two forced-token runs and
prints the first stages whose hashes or compact values diverge.
"""

import argparse
import json
import math
from pathlib import Path


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line"] = lineno
            rows.append(obj)
    return rows


def key(r):
    return (r.get("pos"), r.get("layer"), r.get("name"), r.get("type"))


def numdiff(a, b, field):
    if field not in a or field not in b:
        return None
    try:
        av = float(a[field])
        bv = float(b[field])
    except (TypeError, ValueError):
        return None
    if math.isnan(av) or math.isnan(bv):
        return None
    return bv - av


def fmt_delta(d):
    if d is None:
        return ""
    return f" {d:+.6g}"


def changed(a, b):
    if key(a) != key(b):
        return True
    if a.get("hash") != b.get("hash"):
        return True
    if a.get("type") == "i32" and a.get("values") != b.get("values"):
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--all", action="store_true", help="print all rows, not just changed rows")
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)
    n = min(len(base), len(cand))
    print(f"baseline_rows={len(base)} candidate_rows={len(cand)} aligned_rows={n}")

    shown = 0
    changed_count = 0
    first = None
    for i in range(n):
        a = base[i]
        b = cand[i]
        is_changed = changed(a, b)
        if is_changed:
            changed_count += 1
            if first is None:
                first = i
        if not args.all and not is_changed:
            continue
        if shown >= args.limit:
            continue
        ka = key(a)
        kb = key(b)
        prefix = "!" if is_changed else "="
        if ka != kb:
            print(f"{prefix} row={i} baseline_key={ka} candidate_key={kb}")
        elif a.get("type") == "f32":
            print(
                f"{prefix} row={i} line={a.get('_line')}/{b.get('_line')} "
                f"pos={ka[0]} layer={ka[1]} name={ka[2]} "
                f"hash={a.get('hash')}->{b.get('hash')} "
                f"meanΔ={fmt_delta(numdiff(a,b,'mean'))} "
                f"rmsΔ={fmt_delta(numdiff(a,b,'rms'))} "
                f"max_absΔ={fmt_delta(numdiff(a,b,'max_abs'))} "
                f"idx={a.get('max_abs_idx')}->{b.get('max_abs_idx')}"
            )
        else:
            print(
                f"{prefix} row={i} line={a.get('_line')}/{b.get('_line')} "
                f"pos={ka[0]} layer={ka[1]} name={ka[2]} "
                f"hash={a.get('hash')}->{b.get('hash')} "
                f"values={a.get('values')}->{b.get('values')}"
            )
        shown += 1

    if len(base) != len(cand):
        print(f"length_mismatch baseline_extra={len(base)-n} candidate_extra={len(cand)-n}")
    print(f"changed_rows={changed_count}")
    if first is not None:
        a = base[first]
        b = cand[first]
        print(f"first_changed_row={first} baseline_line={a.get('_line')} candidate_line={b.get('_line')} key={key(a)}")


if __name__ == "__main__":
    main()
