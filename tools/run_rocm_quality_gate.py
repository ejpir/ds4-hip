#!/usr/bin/env python3
"""Run a quality-oriented ROCm gate against old-HIP reference JSON dumps.

This runner intentionally does not use exact selected-token parity as its only
success criterion.  For each prompt it can:
  1. run candidate greedy logprob dump for text/argmax visibility;
  2. run candidate with --force-tokens against an old-HIP reference JSON;
  3. summarize forced-stream logprob deltas with quality_logprob_report.py.

It refuses to run while the old-HIP server PID file is live unless explicitly
allowed, because ROCm validation should not share the GPU with production-safe
old-HIP serving.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = (
    "/home/nick/.cache/huggingface/hub/models--cyberneurova--"
    "CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/"
    "665c8e035e2602d12d28b84920808b158f337e09/"
    "cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
)

CASES = [
    {
        "name": "hello",
        "prompt": "Hello",
        "ctx": 512,
        "n": 57,
        "baseline": "/tmp/contval_hello_hip.json",
    },
    {
        "name": "math",
        "prompt": "What is 2+2?",
        "ctx": 512,
        "n": 36,
        "baseline": "/tmp/contval_math_hip.json",
    },
    {
        "name": "haiku",
        "prompt": "Write a haiku about rain.",
        "ctx": 512,
        "n": 64,
        "baseline": "/tmp/contval_haiku_hip.json",
    },
    {
        "name": "p1741",
        "prompt_file": "/tmp/prompt1741.txt",
        "ctx": 4096,
        "n": 64,
        "baseline": "/tmp/contval_p1741_hip64.json",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def live_pid(pid_path: Path) -> int | None:
    try:
        text = pid_path.read_text().strip()
        pid = int(text)
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def parse_env(kv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in kv:
        if "=" not in item:
            raise SystemExit(f"--candidate-env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if not k:
            raise SystemExit(f"empty env key in {item!r}")
        out[k] = v
    return out


def run(cmd: list[str], *, env: dict[str, str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        proc = subprocess.run(cmd, cwd=repo_root(), env=env, stdout=out, stderr=err)
    return proc.returncode


def run_capture(cmd: list[str], *, env: dict[str, str], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        proc = subprocess.run(cmd, cwd=repo_root(), env=env, stdout=out, stderr=subprocess.STDOUT)
    return proc.returncode


def prompt_args(case: dict[str, object]) -> list[str]:
    if "prompt_file" in case:
        return ["--prompt-file", str(case["prompt_file"])]
    return ["-p", str(case["prompt"])]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run DS4 ROCm quality gate")
    ap.add_argument("--model", default=os.environ.get("DS4_MODEL", DEFAULT_MODEL))
    ap.add_argument("--candidate-bin", default="./ds4-rocm-upstream")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--case", action="append", choices=[c["name"] for c in CASES],
                    help="case name to run; may be repeated; default runs all")
    ap.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--skip-forced", action="store_true")
    ap.add_argument("--allow-server", action="store_true",
                    help="allow running even if /tmp/ds4-server.pid is live")
    ap.add_argument("--greedy-top-k", type=int, default=20)
    ap.add_argument("--forced-top-k", type=int, default=128)
    ap.add_argument("--near-tie-logit-margin", type=float, default=0.25)
    ap.add_argument("--fail-mean-logprob-drop", type=float, default=None)
    ap.add_argument("--fail-worst-logprob-drop", type=float, default=None)
    ap.add_argument("--fail-candidate-missing-frac", type=float, default=None)
    args = ap.parse_args()

    pid = live_pid(Path("/tmp/ds4-server.pid"))
    if pid is not None and not args.allow_server:
        print(
            f"refusing to run ROCm quality gate while old-HIP server is live: pid={pid}\n"
            "stop it first or pass --allow-server if you know the GPU is isolated",
            file=sys.stderr,
        )
        return 2

    cand_bin = repo_root() / args.candidate_bin
    if not cand_bin.exists():
        print(f"candidate binary not found: {args.candidate_bin}", file=sys.stderr)
        return 2
    model = Path(args.model)
    if not model.exists():
        print(f"model not found: {model}", file=sys.stderr)
        return 2

    selected = set(args.case or [c["name"] for c in CASES])
    cases = [c for c in CASES if c["name"] in selected]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"/tmp/ds4_rocm_quality_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_env = parse_env(args.candidate_env)
    base_env = os.environ.copy()
    base_env.update(extra_env)

    report_lines = [
        f"out_dir={out_dir}",
        f"model={model}",
        f"candidate_bin={args.candidate_bin}",
        "candidate_env=" + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(extra_env.items())),
        "",
    ]
    failed = False

    for case in cases:
        name = str(case["name"])
        baseline = Path(str(case["baseline"]))
        if not baseline.exists():
            report_lines.append(f"[{name}] SKIP missing baseline {baseline}")
            failed = True
            continue
        common = [
            args.candidate_bin,
            "-m", str(model),
            "--ctx", str(case["ctx"]),
            *prompt_args(case),
            "--temp", "0",
            "-n", str(case["n"]),
        ]
        report_lines.append(f"[{name}] ctx={case['ctx']} n={case['n']} baseline={baseline}")

        if not args.skip_greedy:
            greedy_json = out_dir / f"{name}_greedy.json"
            greedy_out = out_dir / f"{name}_greedy.out"
            greedy_err = out_dir / f"{name}_greedy.err"
            env = base_env.copy()
            env["DS4_LOCK_FILE"] = f"/tmp/ds4-quality-{name}-greedy.lock"
            cmd = [*common, "--dump-logprobs", str(greedy_json), "--logprobs-top-k", str(args.greedy_top_k)]
            rc = run(cmd, env=env, stdout=greedy_out, stderr=greedy_err)
            report_lines.append(f"  greedy_rc={rc} json={greedy_json}")
            cmp_out = out_dir / f"{name}_greedy_compare.txt"
            cmp_cmd = [
                sys.executable, "tools/compare_logprobs.py", str(baseline), str(greedy_json),
                "--show-top", "5", "--show-text",
            ]
            cmp_rc = run_capture(cmp_cmd, env=base_env, output=cmp_out)
            first = ""
            if cmp_out.exists():
                for line in cmp_out.read_text(errors="replace").splitlines():
                    if "match:" in line or "mismatch_step" in line or "length mismatch" in line:
                        first = line
                        break
            report_lines.append(f"  greedy_compare_rc={cmp_rc} {first} report={cmp_out}")
            if rc != 0:
                failed = True

        if not args.skip_forced:
            forced_json = out_dir / f"{name}_forced_refstream.json"
            forced_out = out_dir / f"{name}_forced_refstream.out"
            forced_err = out_dir / f"{name}_forced_refstream.err"
            env = base_env.copy()
            env["DS4_LOCK_FILE"] = f"/tmp/ds4-quality-{name}-forced.lock"
            cmd = [
                *common,
                "--dump-logprobs", str(forced_json),
                "--logprobs-top-k", str(args.forced_top_k),
                "--force-tokens", str(baseline),
            ]
            rc = run(cmd, env=env, stdout=forced_out, stderr=forced_err)
            report_lines.append(f"  forced_rc={rc} json={forced_json}")
            quality_out = out_dir / f"{name}_quality_report.txt"
            quality_cmd = [
                sys.executable, "tools/quality_logprob_report.py",
                str(baseline), str(forced_json),
                "--baseline-token-field", "selected",
                "--candidate-token-field", "eval",
                "--near-tie-logit-margin", str(args.near_tie_logit_margin),
                "--worst", "8",
            ]
            if args.fail_mean_logprob_drop is not None:
                quality_cmd += ["--fail-mean-logprob-drop", str(args.fail_mean_logprob_drop)]
            if args.fail_worst_logprob_drop is not None:
                quality_cmd += ["--fail-worst-logprob-drop", str(args.fail_worst_logprob_drop)]
            if args.fail_candidate_missing_frac is not None:
                quality_cmd += ["--fail-candidate-missing-frac", str(args.fail_candidate_missing_frac)]
            qrc = run_capture(quality_cmd, env=base_env, output=quality_out)
            summary = []
            if quality_out.exists():
                for line in quality_out.read_text(errors="replace").splitlines():
                    if line.startswith("logprob_delta") or line.startswith("topk_presence") or line.startswith("scored_steps"):
                        summary.append(line)
            report_lines.append(f"  quality_rc={qrc} report={quality_out}")
            for line in summary[:3]:
                report_lines.append(f"    {line}")
            if rc != 0 or qrc != 0:
                failed = True
        report_lines.append("")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print(f"summary={summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
