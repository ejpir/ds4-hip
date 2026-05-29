#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DEFAULT="/home/nick/.cache/huggingface/hub/models--cyberneurova--CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/665c8e035e2602d12d28b84920808b158f337e09/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"

usage() {
  cat <<'EOF'
Usage:
  scripts/start_ds4_cli_rocm_upstream.sh [ds4 CLI args]

Starts the upstream-shaped ROCm interactive CLI with the current fast-full
CyberNeurova/Promessi preset. No -p/--prompt is supplied by default, so it opens
an interactive ds4> prompt.

Defaults:
  DS4_MODEL=$CyberNeurova_Q2_K_GGUF
  DS4_CLI_CTX=131072
  DS4_CLI_TOKENS=1024
  DS4_SERVER_FAST_FULL=1
  DS4_SERVER_PERFLEVEL=high

Common examples:
  scripts/start_ds4_cli_rocm_upstream.sh
  scripts/start_ds4_cli_rocm_upstream.sh --nothink
  DS4_CLI_CTX=8192 DS4_CLI_TOKENS=2048 scripts/start_ds4_cli_rocm_upstream.sh
  scripts/start_ds4_cli_rocm_upstream.sh -p "Write a haiku about GPUs" --nothink

Safety:
  By default this refuses to run while the production ds4-server pidfile process
  is alive. Set DS4_CLI_STOP_SERVER=1 to stop that server first, or
  DS4_CLI_ALLOW_WITH_SERVER=1 to bypass the check.

Useful environment overrides:
  DS4_MODEL=FILE
  DS4_CLI_CTX=N
  DS4_CLI_TOKENS=N
  DS4_CLI_STOP_SERVER=1
  DS4_CLI_ALLOW_WITH_SERVER=1
  DS4_SERVER_ATTENTION_OUTPUT_F16_OUT=0  Disable default FAST_FULL attention-output f16 projection
  DS4_SERVER_ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS=N  Optional attention f16-output minimum; default 0
  DS4_SERVER_SHARED_DOWN_F16_OUT=0       Disable default FAST_FULL shared-down f16 projection
  DS4_SERVER_SHARED_DOWN_F16_OUT_MIN_TOKENS=N       Optional shared-down f16-output minimum; default 0
  DS4_SERVER_Q8_WMMA_4W=1                Experimental 4-wave 64x64 Q8 WMMA path for large prefill batches; FAST_FULL default 1
  DS4_SERVER_Q8_WMMA_4W_MIN_TOKENS=N     Minimum token batch for 4-wave Q8 WMMA; default 256
  DS4_SERVER_Q8_WMMA_4W_MIN_OUT=N        Minimum output rows for 4-wave Q8 WMMA; default 1024
  DS4_SERVER_Q8_PREQUANT_DECODE=0        Disable default FAST_FULL prequantized Q8 decode matvecs
  DS4_SERVER_Q8_DECODE_RPB=1|2|4|8|16|32 Rows/block for single-token Q8 decode matvecs; FAST_FULL default 1
  DS4_SERVER_Q8_HC_DECODE_RPB=1|2|4|8|16|32 Rows/block for fused Q8 HC-expand decode matvecs; FAST_FULL default 16
  DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB=1|2|4|8|16|32 Rows/block for decode attn_output_a low projection; FAST_FULL default 32
  DS4_SERVER_OLDHIP_ATTENTION_DECODE=1  Use the old-HIP decode attention kernel; FAST_FULL default 1
  DS4_SERVER_QKV_PAIR_DECODE=1           Opt in shared-quant Q/KV decode projection pair; FAST_FULL default 1
  DS4_SERVER_OVERLAP_SHARED_GATE_UP=0    Disable FAST_FULL async shared-expert gate/up overlap
  DS4_SERVER_MOE_DECODE_RPB=N            Rows/block for exact single-token Q2_K MoE decode; FAST_FULL default 1
  DS4_SERVER_MOE_DECODE_Q8K_DOWN=1       Opt in Q8_K mid/down MoE decode path; FAST_FULL default 1
  DS4_SERVER_ATTN_OUT_LOW_SPLITK=1       Restore old split-K decode attn_output_a path; FAST_FULL defaults to prequant
  DS4_SERVER_SHARED_GATE_UP_FUSED_W32=1  Restore old fused shared gate/up float-row decode path; FAST_FULL defaults to prequant pair
  DS4_SERVER_FAST_FULL=0       Disable the preset and use your explicit env
  DS4_SERVER_PERFLEVEL=auto    Or empty/off/skip to bypass rocm-smi perflevel changes
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "$ROOT_DIR/scripts/rocm_settings.sh"

arg_present() {
  local want_short="$1"
  local want_long="$2"
  shift 2
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$want_short" || "$arg" == "$want_long" || "$arg" == "$want_long="* ]] && return 0
  done
  return 1
}

# This script is intentionally fast by default. Shared defaults and alias exports
# live in scripts/rocm_settings.sh.
export DS4_SERVER_FAST_FULL="${DS4_SERVER_FAST_FULL:-1}"
export DS4_SERVER_PERFLEVEL="${DS4_SERVER_PERFLEVEL-high}"
if [[ "${DS4_SERVER_FAST_FULL:-0}" == "1" ]]; then
  ds4_rocm_apply_fast_full_defaults 2048
fi

ds4_rocm_apply_tensor_env
ds4_rocm_apply_prefill_env 128
ds4_rocm_export_runtime_env

pidfile="${DS4_SERVER_PID:-/tmp/ds4-server.pid}"
if [[ -f "$pidfile" ]]; then
  oldpid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ "$oldpid" =~ ^[0-9]+$ ]] && kill -0 "$oldpid" 2>/dev/null; then
    if [[ "${DS4_CLI_STOP_SERVER:-0}" == "1" ]]; then
      echo "ds4-cli-rocm-upstream: stopping ds4-server pid $oldpid" >&2
      kill "$oldpid" 2>/dev/null || true
      for _ in {1..40}; do
        kill -0 "$oldpid" 2>/dev/null || break
        sleep 0.2
      done
      kill -0 "$oldpid" 2>/dev/null && kill -9 "$oldpid" 2>/dev/null || true
      rm -f "$pidfile" /tmp/ds4.lock
    elif [[ "${DS4_CLI_ALLOW_WITH_SERVER:-0}" != "1" ]]; then
      echo "ds4-cli-rocm-upstream: refusing to run while ds4-server pid $oldpid is alive" >&2
      echo "Set DS4_CLI_STOP_SERVER=1 to stop it, or DS4_CLI_ALLOW_WITH_SERVER=1 to bypass." >&2
      exit 1
    fi
  fi
fi

if [[ -f /tmp/ds4.lock ]]; then
  lockpid="$(cat /tmp/ds4.lock 2>/dev/null || true)"
  if [[ "$lockpid" =~ ^[0-9]+$ ]] && ! kill -0 "$lockpid" 2>/dev/null; then
    echo "ds4-cli-rocm-upstream: removing stale /tmp/ds4.lock for dead pid $lockpid" >&2
    rm -f /tmp/ds4.lock
  fi
fi

if [[ "${DS4_SERVER_PERFLEVEL:-}" == "off" || "${DS4_SERVER_PERFLEVEL:-}" == "skip" || "${DS4_SERVER_PERFLEVEL:-}" == "none" ]]; then
  DS4_SERVER_PERFLEVEL=""
fi
if [[ -n "${DS4_SERVER_PERFLEVEL:-}" ]] && command -v rocm-smi >/dev/null 2>&1; then
  echo "ds4-cli-rocm-upstream: setting ROCm perflevel ${DS4_SERVER_PERFLEVEL}" >&2
  rocm-smi --setperflevel "${DS4_SERVER_PERFLEVEL}" >&2 || true
fi

make ds4-rocm-upstream -j"$(nproc)"

MODEL="${DS4_MODEL:-$MODEL_DEFAULT}"
if [[ ! -f "$MODEL" ]]; then
  echo "ds4-cli-rocm-upstream: model not found: $MODEL" >&2
  exit 1
fi

args=("$@")
if ! arg_present -m --model "${args[@]}"; then
  args=(-m "$MODEL" "${args[@]}")
fi
if ! arg_present -c --ctx "${args[@]}"; then
  args=(--ctx "${DS4_CLI_CTX:-131072}" "${args[@]}")
fi
if ! arg_present -n --tokens "${args[@]}"; then
  args=(--tokens "${DS4_CLI_TOKENS:-1024}" "${args[@]}")
fi

exec ./ds4-rocm-upstream "${args[@]}"
