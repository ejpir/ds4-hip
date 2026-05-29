#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/rocm_settings.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_rocm_bandwidth_profile.sh [ds4-bench args]

Run ds4-bench under rocprofv3 hardware counters and summarize ROCm-reported
memory traffic/bandwidth per kernel. This is the real FETCH_SIZE/WRITE_SIZE
counter path, not the server's tokens/s log and not the DS4 useful-byte model.

Defaults are chosen for a short decode profile:
  DS4_BENCH_CTX_START=4096 DS4_BENCH_CTX_MAX=4096 DS4_BENCH_GEN_TOKENS=4

Environment:
  DS4_ROCPROF_OUT_DIR=DIR       Output dir. Default /tmp/ds4_rocprof_bw_TIMESTAMP
  DS4_ROCPROF_METRICS="..."     rocprof metrics. Default: FETCH_SIZE
                                On gfx1151, FETCH_SIZE+WRITE_SIZE may exceed
                                the one-pass counter set; run WRITE_SIZE as a
                                separate pass if needed. Useful optional
                                additions if they fit one pass:
                                L2CacheHit MemUnitBusy OccupancyPercent Wavefronts VALUInsts
  DS4_ROCPROF_KERNEL_REGEX=RE   Optional rocprof --kernel-include-regex filter
  DS4_ROCPROF_TOP=N             Summary rows. Default 30

Typical tuned ctx4096 run:
  DS4_SERVER_PERFLEVEL=off \
  DS4_SERVER_FAST_FULL=1 \
  DS4_SERVER_MOE_DECODE_RPB=1 \
  DS4_SERVER_OLDHIP_ATTENTION_DECODE=1 \
  DS4_SERVER_Q8_DECODE_RPB=1 \
  DS4_SERVER_Q8_HC_DECODE_RPB=16 \
  DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB=32 \
  DS4_SERVER_QKV_PAIR_DECODE=1 \
  DS4_SERVER_MOE_DECODE_Q8K_DOWN=1 \
  DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR_SWIGLU=1 \
  scripts/run_rocm_bandwidth_profile.sh

Notes:
  - Stop the live server first, or set DS4_BENCH_ALLOW_WITH_SERVER=1 if you
    intentionally want contention/noisy counters.
  - rocprofv3 may reject metric sets that cannot be collected in one pass on
    gfx1151. Start with FETCH_SIZE WRITE_SIZE, then add counters.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v rocprofv3 >/dev/null 2>&1; then
  echo "run_rocm_bandwidth_profile: rocprofv3 not found" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${DS4_ROCPROF_OUT_DIR:-/tmp/ds4_rocprof_bw_${stamp}}"
prof_dir="$out_dir/rocprof"
mkdir -p "$prof_dir"

metrics_s="${DS4_ROCPROF_METRICS:-FETCH_SIZE}"
# shellcheck disable=SC2206
metrics=( $metrics_s )

export DS4_SERVER_PERFLEVEL="${DS4_SERVER_PERFLEVEL:-off}"
export DS4_BENCH_CTX_START="${DS4_BENCH_CTX_START:-4096}"
export DS4_BENCH_CTX_MAX="${DS4_BENCH_CTX_MAX:-$DS4_BENCH_CTX_START}"
export DS4_BENCH_STEP_INCR="${DS4_BENCH_STEP_INCR:-4096}"
export DS4_BENCH_GEN_TOKENS="${DS4_BENCH_GEN_TOKENS:-4}"
export DS4_BENCH_CSV="${DS4_BENCH_CSV:-$out_dir/bench.csv}"

cmd=(rocprofv3 --pmc "${metrics[@]}" --output-format csv --output-directory "$prof_dir" --output-file counters)
if [[ -n "${DS4_ROCPROF_KERNEL_REGEX:-}" ]]; then
  cmd+=(--kernel-include-regex "$DS4_ROCPROF_KERNEL_REGEX")
fi
cmd+=(-- scripts/run_ds4_bench_rocm_upstream.sh "$@")

{
  echo "out_dir=$out_dir"
  echo "metrics=${metrics[*]}"
  echo "kernel_regex=${DS4_ROCPROF_KERNEL_REGEX:-}"
  echo "bench_ctx=$DS4_BENCH_CTX_START..$DS4_BENCH_CTX_MAX gen=$DS4_BENCH_GEN_TOKENS"
  printf 'cmd='; printf '%q ' "${cmd[@]}"; echo
} | tee "$out_dir/run.info"

"${cmd[@]}" >"$out_dir/run.out" 2>"$out_dir/run.err"

if [[ -f "$DS4_BENCH_CSV" ]]; then
  echo "== bench ==" | tee "$out_dir/summary.txt"
  cat "$DS4_BENCH_CSV" | tee -a "$out_dir/summary.txt"
fi

echo "== rocprof bandwidth ==" | tee -a "$out_dir/summary.txt"
python3 tools/summarize_rocprof_bandwidth.py --top "${DS4_ROCPROF_TOP:-30}" "$prof_dir" | tee -a "$out_dir/summary.txt"

echo "$out_dir"
