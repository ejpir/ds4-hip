#!/usr/bin/env python3
"""Compare ds4 --dump-logprobs greedy selected-token dumps.

Exits 0 when the selected token id sequence matches for the compared prefix,
otherwise prints the first divergence and exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def token_text(tok: dict[str, Any]) -> str:
    if "text" in tok and tok["text"] is not None:
        return repr(tok["text"])
    bs = tok.get("bytes") or []
    try:
        return repr(bytes(bs).decode("utf-8", "replace"))
    except Exception:
        return repr(bs)


def selected_ids(doc: dict[str, Any]) -> list[int]:
    return [int(step["selected"]["id"]) for step in doc.get("steps", [])]


def selected_text(doc: dict[str, Any], limit: int | None = None) -> str:
    out: list[str] = []
    for step in doc.get("steps", [])[:limit]:
        tok = step.get("selected", {})
        if "text" in tok and tok["text"] is not None:
            out.append(str(tok["text"]))
        else:
            out.append(bytes(tok.get("bytes") or []).decode("utf-8", "replace"))
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ds4 --dump-logprobs selected-token JSON dumps")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--steps", type=int, default=0, help="compare at most N steps; default compares common length")
    ap.add_argument("--show-text", action="store_true", help="print generated selected-token text prefixes")
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())
    b_ids = selected_ids(base)
    c_ids = selected_ids(cand)
    n = min(len(b_ids), len(c_ids))
    if args.steps > 0:
        n = min(n, args.steps)

    first = None
    for i in range(n):
        if b_ids[i] != c_ids[i]:
            first = i
            break

    print(f"baseline_steps={len(b_ids)} candidate_steps={len(c_ids)} compared={n}")
    print(f"baseline_prompt_tokens={base.get('prompt_tokens')} candidate_prompt_tokens={cand.get('prompt_tokens')}")

    if first is None and len(b_ids) == len(c_ids) or (first is None and args.steps > 0):
        print(f"match: selected token ids match for {n} step(s)")
        if args.show_text:
            print("text:", selected_text(cand, n))
        return 0

    if first is None:
        first = n
        print(f"length mismatch after matching prefix={n}")
    else:
        b_tok = base["steps"][first]["selected"]
        c_tok = cand["steps"][first]["selected"]
        print(f"mismatch_step={first} matched_prefix={first}")
        print(f"baseline id={b_tok.get('id')} text={token_text(b_tok)} bytes={b_tok.get('bytes')}")
        print(f"candidate id={c_tok.get('id')} text={token_text(c_tok)} bytes={c_tok.get('bytes')}")
    if args.show_text:
        print("baseline_text:", selected_text(base, n))
        print("candidate_text:", selected_text(cand, n))
    return 1


if __name__ == "__main__":
    sys.exit(main())
