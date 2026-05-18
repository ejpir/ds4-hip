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


def step_token(step: dict[str, Any], field: str) -> dict[str, Any]:
    if field != "selected" and field in step:
        return step.get(field, {})
    return step.get("selected", {})


def selected_ids(doc: dict[str, Any], field: str = "selected") -> list[int]:
    return [int(step_token(step, field)["id"]) for step in doc.get("steps", [])]


def selected_text(doc: dict[str, Any], limit: int | None = None, field: str = "selected") -> str:
    out: list[str] = []
    for step in doc.get("steps", [])[:limit]:
        tok = step_token(step, field)
        if "text" in tok and tok["text"] is not None:
            out.append(str(tok["text"]))
        else:
            out.append(bytes(tok.get("bytes") or []).decode("utf-8", "replace"))
    return "".join(out)


def fmt_top_entry(entry: dict[str, Any]) -> str:
    tok = entry.get("token", {})
    logit = entry.get("logit")
    logprob = entry.get("logprob")
    return f"id={tok.get('id')} text={token_text(tok)} logit={logit} logprob={logprob}"


def print_top(label: str, step: dict[str, Any], limit: int) -> None:
    top = step.get("top_logprobs") or []
    if not top or limit <= 0:
        return
    print(f"{label}_top:")
    for i, entry in enumerate(top[:limit]):
        print(f"  {i}: {fmt_top_entry(entry)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ds4 --dump-logprobs selected-token JSON dumps")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--steps", type=int, default=0, help="compare at most N steps; default compares common length")
    ap.add_argument("--show-text", action="store_true", help="print generated selected-token text prefixes")
    ap.add_argument("--show-top", type=int, default=0, help="print top-N logit/logprob rows at the first mismatch")
    ap.add_argument("--token-field", choices=("selected", "eval"), default="selected",
                    help="token object to compare; eval is useful for forced-token dumps")
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())
    b_ids = selected_ids(base, args.token_field)
    c_ids = selected_ids(cand, args.token_field)
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
            print("text:", selected_text(cand, n, args.token_field))
        return 0

    if first is None:
        first = n
        print(f"length mismatch after matching prefix={n}")
    else:
        b_tok = step_token(base["steps"][first], args.token_field)
        c_tok = step_token(cand["steps"][first], args.token_field)
        print(f"mismatch_step={first} matched_prefix={first}")
        print(f"baseline id={b_tok.get('id')} text={token_text(b_tok)} bytes={b_tok.get('bytes')}")
        print(f"candidate id={c_tok.get('id')} text={token_text(c_tok)} bytes={c_tok.get('bytes')}")
        if args.show_top > 0:
            print_top("baseline", base["steps"][first], args.show_top)
            print_top("candidate", cand["steps"][first], args.show_top)
    if args.show_text:
        print("baseline_text:", selected_text(base, n, args.token_field))
        print("candidate_text:", selected_text(cand, n, args.token_field))
    return 1


if __name__ == "__main__":
    sys.exit(main())
