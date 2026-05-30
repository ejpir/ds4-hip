#!/usr/bin/env python3
"""Summarize/compare GGUF tensor quantization layouts for DS4 ROCm tuning.

Examples:
  tools/gguf_quant_summary.py model.gguf
  tools/gguf_quant_summary.py cyber.gguf standard.gguf --diff
"""
from __future__ import annotations

import argparse
import collections
import importlib.machinery
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
gguf = importlib.machinery.SourceFileLoader(
    "gguf_tensor_offsets", os.path.join(HERE, "gguf_tensor_offsets")
).load_module()

ROUTED_PATTERNS = {
    "gate": re.compile(r"(^|\.)ffn_gate_exps\.weight$"),
    "up": re.compile(r"(^|\.)ffn_up_exps\.weight$"),
    "down": re.compile(r"(^|\.)ffn_down_exps\.weight$"),
}
INTERESTING = [
    ("token_embd", re.compile(r"(^|\.)token_embd\.weight$")),
    ("output", re.compile(r"(^|\.)output\.weight$")),
    ("attn_q_a", re.compile(r"attn_q_a\.weight$")),
    ("attn_q_b", re.compile(r"attn_q_b\.weight$")),
    ("attn_kv", re.compile(r"attn_kv\.weight$")),
    ("attn_output_a", re.compile(r"attn_output_a\.weight$")),
    ("attn_output_b", re.compile(r"attn_output_b\.weight$")),
    ("compressor_gate", re.compile(r"attn_compressor_gate\.weight$")),
    ("compressor_kv", re.compile(r"attn_compressor_kv\.weight$")),
    ("indexer_q_b", re.compile(r"indexer\.attn_q_b\.weight$")),
    ("shared_gate", re.compile(r"ffn_gate_shexp\.weight$")),
    ("shared_up", re.compile(r"ffn_up_shexp\.weight$")),
    ("shared_down", re.compile(r"ffn_down_shexp\.weight$")),
    ("routed_gate", ROUTED_PATTERNS["gate"]),
    ("routed_up", ROUTED_PATTERNS["up"]),
    ("routed_down", ROUTED_PATTERNS["down"]),
]


def typ(t: int) -> str:
    return gguf.GGUF_TYPES.get(t, str(t))


def dims_s(dims: Iterable[int]) -> str:
    return "x".join(str(x) for x in dims)


def summarize(path: str):
    version, alignment, tensors = gguf.parse_gguf(path)
    by_name = {t.name: t for t in tensors}
    return version, alignment, tensors, by_name


def print_counter(label: str, c: collections.Counter[int], indent: str = "  ") -> None:
    parts = [f"{typ(k)}={v}" for k, v in sorted(c.items(), key=lambda kv: (typ(kv[0]), kv[0]))]
    print(f"{indent}{label}: " + (", ".join(parts) if parts else "none"))


def summarize_one(path: str) -> None:
    version, alignment, tensors, _ = summarize(path)
    print(f"{path}")
    print(f"  version={version} alignment={alignment} tensors={len(tensors)}")
    print_counter("all types", collections.Counter(t.typ for t in tensors))
    for label, rx in INTERESTING:
        matches = [t for t in tensors if rx.search(t.name)]
        if not matches:
            continue
        c = collections.Counter(t.typ for t in matches)
        shapes = collections.Counter(tuple(t.dims) for t in matches)
        shape_part = ", ".join(f"{dims_s(k)}={v}" for k, v in shapes.most_common())
        print_counter(label, c)
        print(f"    shapes: {shape_part}")
    print()


def diff(a_path: str, b_path: str) -> int:
    _, _, a_tensors, a = summarize(a_path)
    _, _, b_tensors, b = summarize(b_path)
    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print()
    print("Top-level type counts:")
    ac = collections.Counter(t.typ for t in a_tensors)
    bc = collections.Counter(t.typ for t in b_tensors)
    for k in sorted(set(ac) | set(bc), key=lambda x: (typ(x), x)):
        if ac[k] != bc[k]:
            print(f"  {typ(k):8s} A={ac[k]:4d} B={bc[k]:4d} delta={bc[k]-ac[k]:+d}")
    print()

    print("Routed expert quantization:")
    for label, rx in ROUTED_PATTERNS.items():
        ca = collections.Counter(t.typ for t in a_tensors if rx.search(t.name))
        cb = collections.Counter(t.typ for t in b_tensors if rx.search(t.name))
        print(f"  {label}:")
        print(f"    A: " + (", ".join(f"{typ(k)}={v}" for k, v in sorted(ca.items())) or "none"))
        print(f"    B: " + (", ".join(f"{typ(k)}={v}" for k, v in sorted(cb.items())) or "none"))
    print()

    common = sorted(set(a) & set(b))
    type_diffs = []
    dim_diffs = []
    for name in common:
        if a[name].typ != b[name].typ:
            type_diffs.append((name, typ(a[name].typ), typ(b[name].typ), dims_s(a[name].dims), dims_s(b[name].dims)))
        elif a[name].dims != b[name].dims:
            dim_diffs.append((name, typ(a[name].typ), dims_s(a[name].dims), dims_s(b[name].dims)))

    print(f"Names: common={len(common)} only_A={len(set(a)-set(b))} only_B={len(set(b)-set(a))}")
    if type_diffs:
        print(f"\nTensor type diffs ({len(type_diffs)}):")
        for row in type_diffs[:200]:
            print(f"  {row[0]}: A={row[1]} {row[3]}  B={row[2]} {row[4]}")
        if len(type_diffs) > 200:
            print(f"  ... {len(type_diffs)-200} more")
    if dim_diffs:
        print(f"\nTensor shape diffs ({len(dim_diffs)}):")
        for row in dim_diffs[:100]:
            print(f"  {row[0]} ({row[1]}): A={row[2]} B={row[3]}")
        if len(dim_diffs) > 100:
            print(f"  ... {len(dim_diffs)-100} more")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="one GGUF to summarize, or two GGUFs to compare")
    ap.add_argument("--diff", action="store_true", help="compare exactly two models")
    args = ap.parse_args()
    if args.diff:
        if len(args.models) != 2:
            ap.error("--diff requires exactly two models")
        return diff(args.models[0], args.models[1])
    for p in args.models:
        summarize_one(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
