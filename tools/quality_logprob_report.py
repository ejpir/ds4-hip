#!/usr/bin/env python3
"""Tolerance-oriented ds4 logprob comparison for backend quality gates.

Unlike compare_logprobs.py, this tool is not an exact-token parity gate.  It is
intended for forced-token quality checks: generate a reference continuation with
one backend, run another backend with --force-tokens against that JSON, then
compare the logprob assigned to the same evaluated token stream.

It also prints the first greedy selected-token mismatch when present, but exits
success unless explicit --fail-* thresholds are exceeded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


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
        return step.get(field, {}) or {}
    return step.get("selected", {}) or {}


def token_id(step: dict[str, Any], field: str) -> int | None:
    tok = step_token(step, field)
    try:
        return int(tok["id"])
    except Exception:
        return None


def selected_text(doc: dict[str, Any], limit: int | None = None, field: str = "selected") -> str:
    out: list[str] = []
    steps = doc.get("steps", [])
    if limit is not None:
        steps = steps[:limit]
    for step in steps:
        tok = step_token(step, field)
        if "text" in tok and tok["text"] is not None:
            out.append(str(tok["text"]))
        else:
            out.append(bytes(tok.get("bytes") or []).decode("utf-8", "replace"))
    return "".join(out)


def top_entries(step: dict[str, Any]) -> list[dict[str, Any]]:
    return list(step.get("top_logprobs") or [])


def find_top(step: dict[str, Any], tid: int) -> tuple[int | None, float | None, float | None]:
    for rank, entry in enumerate(top_entries(step)):
        tok = entry.get("token", {}) or {}
        try:
            eid = int(tok.get("id"))
        except Exception:
            continue
        if eid == tid:
            logit = entry.get("logit")
            logprob = entry.get("logprob")
            return rank, float(logit) if logit is not None else None, float(logprob) if logprob is not None else None
    return None, None, None


def top_margin(step: dict[str, Any]) -> float | None:
    top = top_entries(step)
    if len(top) < 2:
        return None
    try:
        return float(top[0]["logit"]) - float(top[1]["logit"])
    except Exception:
        return None


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else float("nan")


def rms(xs: Iterable[float]) -> float:
    vals = list(xs)
    return math.sqrt(sum(x * x for x in vals) / len(vals)) if vals else float("nan")


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def fmt_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.6g}"


def fmt_top(step: dict[str, Any], limit: int) -> list[str]:
    rows = []
    for i, entry in enumerate(top_entries(step)[:limit]):
        tok = entry.get("token", {}) or {}
        rows.append(
            f"  {i}: id={tok.get('id')} text={token_text(tok)} "
            f"logit={entry.get('logit')} logprob={entry.get('logprob')}"
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Quality-oriented ds4 logprob report")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--baseline-token-field", choices=("selected", "eval"), default="selected")
    ap.add_argument("--candidate-token-field", choices=("selected", "eval"), default="eval")
    ap.add_argument("--steps", type=int, default=0, help="compare at most N common steps")
    ap.add_argument("--near-tie-logit-margin", type=float, default=0.25,
                    help="count top-1/top-2 logit margins at or below this value")
    ap.add_argument("--worst", type=int, default=5, help="show N worst candidate logprob drops")
    ap.add_argument("--show-top", type=int, default=0, help="show top-N rows at first selected-token mismatch")
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("--fail-mean-logprob-drop", type=float, default=None,
                    help="exit nonzero if mean(candidate-base logprob) < -VALUE")
    ap.add_argument("--fail-worst-logprob-drop", type=float, default=None,
                    help="exit nonzero if worst(candidate-base logprob) < -VALUE")
    ap.add_argument("--fail-candidate-missing-frac", type=float, default=None,
                    help="exit nonzero if evaluated token is absent from candidate top-k above this fraction")
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())
    b_steps = list(base.get("steps", []))
    c_steps = list(cand.get("steps", []))
    n = min(len(b_steps), len(c_steps))
    if args.steps > 0:
        n = min(n, args.steps)

    print(f"baseline_steps={len(b_steps)} candidate_steps={len(c_steps)} compared={n}")
    print(f"baseline_prompt_tokens={base.get('prompt_tokens')} candidate_prompt_tokens={cand.get('prompt_tokens')}")
    print(f"baseline_field={args.baseline_token_field} candidate_field={args.candidate_token_field}")

    first_selected_mismatch: int | None = None
    for i in range(n):
        if token_id(b_steps[i], "selected") != token_id(c_steps[i], "selected"):
            first_selected_mismatch = i
            break
    if first_selected_mismatch is None:
        print(f"greedy_selected_match_prefix={n}")
    else:
        i = first_selected_mismatch
        bsel = step_token(b_steps[i], "selected")
        csel = step_token(c_steps[i], "selected")
        print(f"greedy_selected_mismatch_step={i} matched_prefix={i}")
        print(f"baseline_selected id={bsel.get('id')} text={token_text(bsel)}")
        print(f"candidate_selected id={csel.get('id')} text={token_text(csel)}")
        if args.show_top > 0:
            print("baseline_top:")
            print("\n".join(fmt_top(b_steps[i], args.show_top)))
            print("candidate_top:")
            print("\n".join(fmt_top(c_steps[i], args.show_top)))

    target_mismatches = 0
    missing_base = 0
    missing_cand = 0
    both = 0
    lp_deltas: list[float] = []
    logit_deltas: list[float] = []
    worst_rows: list[tuple[float, int, int, str, int | None, int | None, float | None, float | None]] = []
    near_ties = 0
    margins: list[float] = []

    for i in range(n):
        btok = step_token(b_steps[i], args.baseline_token_field)
        ctok = step_token(c_steps[i], args.candidate_token_field)
        bid = token_id(b_steps[i], args.baseline_token_field)
        cid = token_id(c_steps[i], args.candidate_token_field)
        if bid is None or cid is None:
            continue
        if bid != cid:
            target_mismatches += 1
        # For forced-token quality reports, compare the baseline target token in
        # both distributions.  If fields differ, the target mismatch is reported
        # but the baseline id remains the scoring target.
        tid = bid
        brank, blogit, blogprob = find_top(b_steps[i], tid)
        crank, clogit, clogprob = find_top(c_steps[i], tid)
        if blogprob is None:
            missing_base += 1
        if clogprob is None:
            missing_cand += 1
        cm = top_margin(c_steps[i])
        if cm is not None:
            margins.append(cm)
            if cm <= args.near_tie_logit_margin:
                near_ties += 1
        if blogprob is not None and clogprob is not None:
            both += 1
            dlp = clogprob - blogprob
            lp_deltas.append(dlp)
            if blogit is not None and clogit is not None:
                logit_deltas.append(clogit - blogit)
            worst_rows.append((dlp, i, tid, token_text(btok), brank, crank, blogprob, clogprob))

    missing_frac = (missing_cand / n) if n else float("nan")
    print(f"eval_token_mismatches={target_mismatches}")
    print(f"topk_presence baseline_missing={missing_base}/{n} candidate_missing={missing_cand}/{n} candidate_missing_frac={fmt_float(missing_frac)}")
    print(f"scored_steps={both}/{n}")
    if lp_deltas:
        print(
            "logprob_delta(candidate-base): "
            f"mean={fmt_float(mean(lp_deltas))} "
            f"median={fmt_float(statistics.median(lp_deltas))} "
            f"p05={fmt_float(percentile(lp_deltas, 0.05))} "
            f"p95={fmt_float(percentile(lp_deltas, 0.95))} "
            f"rms={fmt_float(rms(lp_deltas))} "
            f"worst={fmt_float(min(lp_deltas))} best={fmt_float(max(lp_deltas))}"
        )
        print(f"total_logprob_delta={fmt_float(sum(lp_deltas))} mean_nll_delta(base-candidate)={fmt_float(-mean(lp_deltas))}")
    if logit_deltas:
        print(
            "target_logit_delta(candidate-base): "
            f"mean={fmt_float(mean(logit_deltas))} "
            f"rms={fmt_float(rms(logit_deltas))} "
            f"worst={fmt_float(min(logit_deltas))} best={fmt_float(max(logit_deltas))}"
        )
    if margins:
        print(
            f"candidate_near_ties margin<={args.near_tie_logit_margin:g}: {near_ties}/{len(margins)} "
            f"mean_margin={fmt_float(mean(margins))} min_margin={fmt_float(min(margins))}"
        )

    if args.worst > 0 and worst_rows:
        print("worst_logprob_drops:")
        for dlp, i, tid, text, brank, crank, blogprob, clogprob in sorted(worst_rows)[:args.worst]:
            print(
                f"  step={i} id={tid} text={text} delta={fmt_float(dlp)} "
                f"base_lp={fmt_float(blogprob if blogprob is not None else float('nan'))} "
                f"cand_lp={fmt_float(clogprob if clogprob is not None else float('nan'))} "
                f"base_rank={brank} cand_rank={crank}"
            )

    if args.show_text:
        print("baseline_text:", selected_text(base, n, args.baseline_token_field))
        print("candidate_text:", selected_text(cand, n, args.candidate_token_field))

    failed = False
    if args.fail_candidate_missing_frac is not None and n and missing_frac > args.fail_candidate_missing_frac:
        print(f"FAIL: candidate_missing_frac {missing_frac:.6g} > {args.fail_candidate_missing_frac:.6g}", file=sys.stderr)
        failed = True
    if args.fail_mean_logprob_drop is not None and lp_deltas and mean(lp_deltas) < -args.fail_mean_logprob_drop:
        print(
            f"FAIL: mean logprob delta {mean(lp_deltas):.6g} < -{args.fail_mean_logprob_drop:.6g}",
            file=sys.stderr,
        )
        failed = True
    if args.fail_worst_logprob_drop is not None and lp_deltas and min(lp_deltas) < -args.fail_worst_logprob_drop:
        print(
            f"FAIL: worst logprob delta {min(lp_deltas):.6g} < -{args.fail_worst_logprob_drop:.6g}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
