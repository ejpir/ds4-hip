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
  DS4_SERVER_Q8_PREQUANT_DECODE=0        Disable default FAST_FULL prequantized Q8 decode matvecs
  DS4_SERVER_Q8_DECODE_RPB=1|2|4|8|16|32 Rows/block for single-token Q8 decode matvecs
  DS4_SERVER_Q8_HC_DECODE_RPB=1|2|4|8|16|32 Rows/block for fused Q8 HC-expand decode matvecs
  DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB=1|2|4|8|16|32 Rows/block for decode attn_output_a low projection
  DS4_SERVER_OLDHIP_ATTENTION_DECODE=1  Use the old-HIP decode attention kernel
  DS4_SERVER_QKV_PAIR_DECODE=1           Opt in shared-quant Q/KV decode projection pair
  DS4_SERVER_MOE_DECODE_Q8K_DOWN=1       Opt in Q8_K mid/down MoE decode path
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

export_pair_flag() {
  local suffix="$1"
  local value="${2:-0}"
  if [[ "$value" == "1" ]]; then
    export "DS4_CUDA_${suffix}=1"
    export "DS4_HIP_${suffix}=1"
  fi
}

export_pair_value() {
  local suffix="$1"
  local value="${2:-}"
  if [[ -n "$value" ]]; then
    export "DS4_CUDA_${suffix}=$value"
    export "DS4_HIP_${suffix}=$value"
  fi
}

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

# This script is intentionally fast by default.
export DS4_SERVER_FAST_FULL="${DS4_SERVER_FAST_FULL:-1}"
export DS4_SERVER_PERFLEVEL="${DS4_SERVER_PERFLEVEL-high}"

if [[ "${DS4_SERVER_FAST_FULL:-0}" == "1" ]]; then
  export DS4_SERVER_PREFILL_CHUNK="${DS4_SERVER_PREFILL_CHUNK:-2048}"
  export DS4_SERVER_DEVICE_TENSORS="${DS4_SERVER_DEVICE_TENSORS:-1}"
  export DS4_SERVER_COPY_MODEL="${DS4_SERVER_COPY_MODEL:-1}"
  export DS4_SERVER_COPY_MODEL_CHUNK_MB="${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}"
  export DS4_SERVER_PREFILL_RAW_FAST="${DS4_SERVER_PREFILL_RAW_FAST:-1}"
  export DS4_SERVER_PREFILL_MIXED_FAST="${DS4_SERVER_PREFILL_MIXED_FAST:-1}"
  export DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP="${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-4}"
  export DS4_SERVER_INDEXED_HEADS32="${DS4_SERVER_INDEXED_HEADS32:-1}"
  export DS4_SERVER_Q8_BATCH_FAST="${DS4_SERVER_Q8_BATCH_FAST:-1}"
  export DS4_SERVER_Q8_BATCH_SHARED_X="${DS4_SERVER_Q8_BATCH_SHARED_X:-1}"
  export DS4_SERVER_Q8_BATCH_TILE="${DS4_SERVER_Q8_BATCH_TILE:-32}"
  export DS4_SERVER_Q8_BATCH_RPB="${DS4_SERVER_Q8_BATCH_RPB:-32}"
  export DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS="${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-32}"
  export DS4_SERVER_Q8_GROUPED_BATCH_TILE="${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-32}"
  export DS4_SERVER_MOE_EXPERT_BATCH="${DS4_SERVER_MOE_EXPERT_BATCH:-1}"
  export DS4_SERVER_MOE_GATE_TILE="${DS4_SERVER_MOE_GATE_TILE:-4}"
  export DS4_SERVER_MOE_DOWN_TILE="${DS4_SERVER_MOE_DOWN_TILE:-4}"
  export DS4_SERVER_MOE_GATE_RPB="${DS4_SERVER_MOE_GATE_RPB:-16}"
  export DS4_SERVER_MOE_DOWN_RPB="${DS4_SERVER_MOE_DOWN_RPB:-16}"
  export DS4_SERVER_MOE_EXPERT_SHARED_X="${DS4_SERVER_MOE_EXPERT_SHARED_X:-1}"
  export DS4_SERVER_MOE_EXPERT_SHARED_MID="${DS4_SERVER_MOE_EXPERT_SHARED_MID:-1}"
  export DS4_SERVER_Q8_REPACK="${DS4_SERVER_Q8_REPACK:-0}"
  export DS4_SERVER_Q8_REPACK_SPLIT16="${DS4_SERVER_Q8_REPACK_SPLIT16:-0}"
  export DS4_SERVER_Q8_WMMA_FAST="${DS4_SERVER_Q8_WMMA_FAST:-1}"
  export DS4_SERVER_Q8_PREQUANT_DECODE="${DS4_SERVER_Q8_PREQUANT_DECODE:-1}"
  export DS4_SERVER_ATTN_OUT_LOW_SPLITK="${DS4_SERVER_ATTN_OUT_LOW_SPLITK:-0}"
  export DS4_SERVER_SHARED_GATE_UP_FUSED_W32="${DS4_SERVER_SHARED_GATE_UP_FUSED_W32:-0}"
  export DS4_SERVER_MOE_WMMA_HOT="${DS4_SERVER_MOE_WMMA_HOT:-1}"
  export DS4_SERVER_MOE_WMMA_GATE_HOT="${DS4_SERVER_MOE_WMMA_GATE_HOT:-8}"
  export DS4_SERVER_MOE_WMMA_DOWN_HOT="${DS4_SERVER_MOE_WMMA_DOWN_HOT:-8}"
  export DS4_SERVER_MOE_WMMA_MTILES="${DS4_SERVER_MOE_WMMA_MTILES:-16}"
  export DS4_SERVER_MOE_WMMA_F16_DOWN_ALL="${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-1}"
  export DS4_SERVER_MOE_WMMA_F16_SPLIT="${DS4_SERVER_MOE_WMMA_F16_SPLIT:-1}"
  export DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN="${DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN:-64}"
  export DS4_SERVER_MOE_WMMA_X_F16="${DS4_SERVER_MOE_WMMA_X_F16:-1}"

  # Upstream-shaped ROCm-only extras from the fastest CLI/bench recipe.
  export DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED="${DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED:-1}"
  export DS4_CUDA_SHARED_DOWN_CUBLAS="${DS4_CUDA_SHARED_DOWN_CUBLAS:-1}"
  export DS4_SERVER_ATTN_Q_B_CUBLAS="${DS4_SERVER_ATTN_Q_B_CUBLAS:-1}"
  export DS4_SERVER_ATTN_Q_B_PRELOAD="${DS4_SERVER_ATTN_Q_B_PRELOAD:-1}"
  export DS4_SERVER_ATTN_Q_B_F16_OUT="${DS4_SERVER_ATTN_Q_B_F16_OUT:-1}"
  # The HC split-stride fix makes the f16-output projection shortcuts safe for
  # the fast ROCm preset in short natural probes; keep explicit kill switches.
  export DS4_SERVER_ATTENTION_OUTPUT_F16_OUT="${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT:-1}"
  export DS4_SERVER_SHARED_DOWN_F16_OUT="${DS4_SERVER_SHARED_DOWN_F16_OUT:-1}"
  export DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL="${DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL:-1}"
  export DS4_CUDA_ATTENTION_OUTPUT_PACKED_B_CUBLAS="${DS4_CUDA_ATTENTION_OUTPUT_PACKED_B_CUBLAS:-1}"
  export DS4_CUDA_ATTENTION_OUTPUT_INTERLEAVED_B_CUBLAS="${DS4_CUDA_ATTENTION_OUTPUT_INTERLEAVED_B_CUBLAS:-1}"
fi

if [[ "${DS4_SERVER_DEVICE_TENSORS:-0}" != "1" ]]; then
  export DS4_HIP_MANAGED_TENSORS=1
else
  unset DS4_HIP_MANAGED_TENSORS || true
fi

if [[ -n "${DS4_SERVER_PREFILL_CHUNK:-}" ]]; then
  export DS4_METAL_PREFILL_CHUNK="$DS4_SERVER_PREFILL_CHUNK"
  export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$DS4_SERVER_PREFILL_CHUNK}"
elif [[ "${DS4_SERVER_DEVICE_TENSORS:-0}" == "1" ]]; then
  export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-128}"
  export DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-$DS4_SESSION_PROGRESS_CHUNK_TOKENS}"
fi

export_pair_flag PREFILL_RAW_FAST "${DS4_SERVER_PREFILL_RAW_FAST:-0}"
export_pair_flag PREFILL_MIXED_FAST "${DS4_SERVER_PREFILL_MIXED_FAST:-0}"
export_pair_value ATTENTION_INDEXED_FUSED_VALUE_GROUP "${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-}"
export_pair_value ATTENTION_INDEXED_SPLIT_VALUE_GROUP "${DS4_SERVER_ATTENTION_INDEXED_SPLIT_VALUE_GROUP:-}"
export_pair_flag INDEXED_HEADS32 "${DS4_SERVER_INDEXED_HEADS32:-0}"
export_pair_flag ATTN_Q_B_CUBLAS "${DS4_SERVER_ATTN_Q_B_CUBLAS:-0}"
export_pair_flag ATTN_Q_B_PRELOAD "${DS4_SERVER_ATTN_Q_B_PRELOAD:-0}"
export_pair_flag ATTN_Q_B_F16_OUT "${DS4_SERVER_ATTN_Q_B_F16_OUT:-0}"
export_pair_flag ATTENTION_OUTPUT_F16_OUT "${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT:-0}"
export_pair_value ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS "${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS:-}"
export_pair_flag SHARED_DOWN_F16_OUT "${DS4_SERVER_SHARED_DOWN_F16_OUT:-0}"
export_pair_value SHARED_DOWN_F16_OUT_MIN_TOKENS "${DS4_SERVER_SHARED_DOWN_F16_OUT_MIN_TOKENS:-}"
export_pair_flag Q8_BATCH_FAST "${DS4_SERVER_Q8_BATCH_FAST:-0}"
export_pair_value Q8_BATCH_TILE "${DS4_SERVER_Q8_BATCH_TILE:-}"
export_pair_value Q8_BATCH_RPB "${DS4_SERVER_Q8_BATCH_RPB:-}"
export_pair_flag Q8_BATCH_SHARED_X "${DS4_SERVER_Q8_BATCH_SHARED_X:-0}"
export_pair_value Q8_BATCH_SHARED_X_BLOCKS "${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-}"
export_pair_value Q8_GROUPED_BATCH_TILE "${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-}"
export_pair_flag Q8_REPACK "${DS4_SERVER_Q8_REPACK:-0}"
export_pair_flag Q8_REPACK_SPLIT16 "${DS4_SERVER_Q8_REPACK_SPLIT16:-0}"
export_pair_flag Q8_WMMA_FAST "${DS4_SERVER_Q8_WMMA_FAST:-0}"
export_pair_value Q8_DECODE_RPB "${DS4_SERVER_Q8_DECODE_RPB:-}"
export_pair_value Q8_HC_DECODE_RPB "${DS4_SERVER_Q8_HC_DECODE_RPB:-}"
export_pair_value ATTN_OUT_LOW_DECODE_RPB "${DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB:-}"
export_pair_flag OLDHIP_ATTENTION_DECODE "${DS4_SERVER_OLDHIP_ATTENTION_DECODE:-0}"
export_pair_flag QKV_PAIR_DECODE "${DS4_SERVER_QKV_PAIR_DECODE:-0}"
if [[ "${DS4_SERVER_Q8_PREQUANT_DECODE:-0}" == "1" ]]; then
  export DS4_CUDA_Q8_PREQUANT_DECODE=1
elif [[ -n "${DS4_SERVER_Q8_PREQUANT_DECODE:-}" ]]; then
  unset DS4_CUDA_Q8_PREQUANT_DECODE || true
fi
if [[ "${DS4_SERVER_ATTN_OUT_LOW_SPLITK:-0}" == "1" ]]; then
  unset DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW || true
else
  export DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW=1
fi
if [[ "${DS4_SERVER_SHARED_GATE_UP_FUSED_W32:-1}" == "1" ]]; then
  unset DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32 || true
else
  export DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32=1
fi
export_pair_flag MOE_EXPERT_BATCH "${DS4_SERVER_MOE_EXPERT_BATCH:-0}"
export_pair_value MOE_EXPERT_TILE "${DS4_SERVER_MOE_EXPERT_TILE:-}"
export_pair_value MOE_GATE_TILE "${DS4_SERVER_MOE_GATE_TILE:-}"
export_pair_value MOE_DOWN_TILE "${DS4_SERVER_MOE_DOWN_TILE:-}"
export_pair_value MOE_GATE_RPB "${DS4_SERVER_MOE_GATE_RPB:-}"
export_pair_value MOE_DOWN_RPB "${DS4_SERVER_MOE_DOWN_RPB:-}"
export_pair_value MOE_DECODE_RPB "${DS4_SERVER_MOE_DECODE_RPB:-}"
export_pair_flag MOE_DECODE_Q8K_DOWN "${DS4_SERVER_MOE_DECODE_Q8K_DOWN:-0}"
export_pair_flag MOE_EXPERT_SHARED_X "${DS4_SERVER_MOE_EXPERT_SHARED_X:-0}"
export_pair_flag MOE_EXPERT_SHARED_MID "${DS4_SERVER_MOE_EXPERT_SHARED_MID:-0}"
export_pair_flag MOE_WMMA_HOT "${DS4_SERVER_MOE_WMMA_HOT:-0}"
export_pair_value MOE_WMMA_GATE_HOT "${DS4_SERVER_MOE_WMMA_GATE_HOT:-}"
export_pair_value MOE_WMMA_DOWN_HOT "${DS4_SERVER_MOE_WMMA_DOWN_HOT:-}"
export_pair_value MOE_WMMA_LAYERS "${DS4_SERVER_MOE_WMMA_LAYERS:-}"
export_pair_value MOE_WMMA_MTILES "${DS4_SERVER_MOE_WMMA_MTILES:-}"
export_pair_flag MOE_WMMA_F16_DOWN_ALL "${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-0}"
export_pair_flag MOE_WMMA_F16_SPLIT "${DS4_SERVER_MOE_WMMA_F16_SPLIT:-0}"
export_pair_value MOE_WMMA_F16_SPLIT_MIN "${DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN:-}"
export_pair_flag MOE_WMMA_X_F16 "${DS4_SERVER_MOE_WMMA_X_F16:-0}"
export_pair_flag COPY_MODEL "${DS4_SERVER_COPY_MODEL:-0}"
export_pair_value COPY_MODEL_CHUNK_MB "${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}"
export_pair_flag CACHE_FINAL_Q8 "${DS4_SERVER_CACHE_FINAL_Q8:-0}"

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
