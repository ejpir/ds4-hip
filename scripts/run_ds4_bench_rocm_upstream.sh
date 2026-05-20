#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DEFAULT="/home/nick/.cache/huggingface/hub/models--cyberneurova--CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/665c8e035e2602d12d28b84920808b158f337e09/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"

usage() {
  cat <<'EOF'
Usage:
  DS4_SERVER_FAST_FULL=1 scripts/run_ds4_bench_rocm_upstream.sh [ds4-bench args]

If no ds4-bench args are supplied, defaults are:
  -m $DS4_MODEL
  --prompt-file speed-bench/promessi_sposi.txt
  --ctx-start 2048 --ctx-max 16384 --step-incr 2048 --gen-tokens 32
  --csv /tmp/ds4_bench_rocm_upstream_fast_full_ctx_TIMESTAMP.csv

Environment:
  DS4_MODEL=FILE                         GGUF model path
  DS4_SERVER_FAST_FULL=1                 Apply the same fast-full preset shape as the server launcher
  DS4_SERVER_PERFLEVEL=high|auto         rocm-smi perflevel; defaults to high
  DS4_BENCH_ALLOW_WITH_SERVER=1          Allow running while ds4-server pidfile process is alive
  DS4_BENCH_PROMPT_FILE=FILE             Default prompt file
  DS4_BENCH_CTX_START=N                  Default ctx-start; default 2048
  DS4_BENCH_CTX_MAX=N                    Default ctx-max; default 16384
  DS4_BENCH_STEP_INCR=N                  Default step-incr; default 2048
  DS4_BENCH_GEN_TOKENS=N                 Default gen-tokens; default 32
  DS4_BENCH_CSV=FILE                     Default CSV output path
  DS4_CUDA_MOE_PROFILE=1                 Profile MoE stages in the upstream-shaped backend
  DS4_CUDA_INDEXED_HEADS32=1             Prototype 32-head indexed-attention block on gfx1151
  DS4_CUDA_ATTN_Q_B_CUBLAS=1             Use f16 GEMM/cache for the large q_b projection
  DS4_CUDA_ATTN_Q_B_PRELOAD=1            Preload q_b Q8_0 weights into the f16 cache
  DS4_CUDA_MOE_WMMA_MTILES=4|8|16        Hot MoE WMMA token tiles/block
  DS4_CUDA_MOE_WMMA_F16_MID=1            Hot MoE WMMA stores/reads routed mid as f16
  DS4_CUDA_MOE_WMMA_F16_MID_ALL=1        Store scalar/cold routed mid scratch as f16 too
  DS4_CUDA_MOE_WMMA_F16_DOWN=1           Hot MoE WMMA writes hot down as f16 and mixed-sums
  DS4_CUDA_MOE_WMMA_F16_DOWN_ALL=1       Store all routed MoE down scratch as f16
  DS4_CUDA_MOE_WMMA_DIRECT_SUM=1         Atomic-add routed MoE down directly to output
  DS4_CUDA_MOE_DENSE_HOT=1               Prototype dense-hot hipBLASLt MoE path
  DS4_CUDA_MOE_DENSE_HOT_TOP=1|2|4       Number of hottest experts/layer to route through dense-hot

All DS4_SERVER_* fast-full knobs are translated to both DS4_CUDA_* and DS4_HIP_*
where the backend has historical aliases, so the upstream-shaped ROCm binary gets
what the server preset intended.
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

# One-command max-performance profile. Keep this intentionally aligned with
# scripts/start_ds4_server.sh, but export CUDA aliases too for ds4_cuda.cu.
export DS4_SERVER_PERFLEVEL="${DS4_SERVER_PERFLEVEL:-high}"

if [[ "${DS4_SERVER_FAST_FULL:-0}" == "1" ]]; then
  export DS4_SERVER_PREFILL_CHUNK="${DS4_SERVER_PREFILL_CHUNK:-2048}"
  export DS4_SERVER_DEVICE_TENSORS="${DS4_SERVER_DEVICE_TENSORS:-1}"
  export DS4_SERVER_COPY_MODEL="${DS4_SERVER_COPY_MODEL:-1}"
  export DS4_SERVER_COPY_MODEL_CHUNK_MB="${DS4_SERVER_COPY_MODEL_CHUNK_MB:-1024}"
  export DS4_SERVER_PREFILL_RAW_FAST="${DS4_SERVER_PREFILL_RAW_FAST:-1}"
  export DS4_SERVER_PREFILL_MIXED_FAST="${DS4_SERVER_PREFILL_MIXED_FAST:-1}"
  export DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP="${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-4}"
  export DS4_SERVER_INDEXED_HEADS32="${DS4_SERVER_INDEXED_HEADS32:-1}"
  export DS4_SERVER_Q8_BATCH_FAST="${DS4_SERVER_Q8_BATCH_FAST:-1}"
  export DS4_SERVER_Q8_BATCH_SHARED_X="${DS4_SERVER_Q8_BATCH_SHARED_X:-1}"
  export DS4_SERVER_Q8_BATCH_TILE="${DS4_SERVER_Q8_BATCH_TILE:-32}"
  export DS4_SERVER_Q8_BATCH_RPB="${DS4_SERVER_Q8_BATCH_RPB:-32}"
  export DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS="${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-16}"
  export DS4_SERVER_Q8_GROUPED_BATCH_TILE="${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-32}"
  export DS4_SERVER_MOE_EXPERT_BATCH="${DS4_SERVER_MOE_EXPERT_BATCH:-1}"
  export DS4_SERVER_MOE_GATE_TILE="${DS4_SERVER_MOE_GATE_TILE:-4}"
  export DS4_SERVER_MOE_DOWN_TILE="${DS4_SERVER_MOE_DOWN_TILE:-4}"
  export DS4_SERVER_MOE_GATE_RPB="${DS4_SERVER_MOE_GATE_RPB:-16}"
  export DS4_SERVER_MOE_DOWN_RPB="${DS4_SERVER_MOE_DOWN_RPB:-16}"
  export DS4_SERVER_MOE_EXPERT_SHARED_X="${DS4_SERVER_MOE_EXPERT_SHARED_X:-1}"
  export DS4_SERVER_MOE_EXPERT_SHARED_MID="${DS4_SERVER_MOE_EXPERT_SHARED_MID:-1}"
  export DS4_SERVER_Q8_REPACK="${DS4_SERVER_Q8_REPACK:-1}"
  export DS4_SERVER_Q8_REPACK_SPLIT16="${DS4_SERVER_Q8_REPACK_SPLIT16:-1}"
  export DS4_SERVER_Q8_WMMA_FAST="${DS4_SERVER_Q8_WMMA_FAST:-1}"
  export DS4_SERVER_MOE_WMMA_HOT="${DS4_SERVER_MOE_WMMA_HOT:-1}"
  export DS4_SERVER_MOE_WMMA_GATE_HOT="${DS4_SERVER_MOE_WMMA_GATE_HOT:-32}"
  export DS4_SERVER_MOE_WMMA_DOWN_HOT="${DS4_SERVER_MOE_WMMA_DOWN_HOT:-32}"

  # Upstream-shaped extras that are not consumed by the old HIP server path but
  # were part of the fastest ROCm CLI recipe.
  export DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED="${DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED:-1}"
  export DS4_CUDA_SHARED_DOWN_CUBLAS="${DS4_CUDA_SHARED_DOWN_CUBLAS:-1}"
  export DS4_SERVER_ATTN_Q_B_CUBLAS="${DS4_SERVER_ATTN_Q_B_CUBLAS:-1}"
  export DS4_SERVER_ATTN_Q_B_PRELOAD="${DS4_SERVER_ATTN_Q_B_PRELOAD:-1}"
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
export_pair_flag Q8_BATCH_FAST "${DS4_SERVER_Q8_BATCH_FAST:-0}"
export_pair_value Q8_BATCH_TILE "${DS4_SERVER_Q8_BATCH_TILE:-}"
export_pair_value Q8_BATCH_RPB "${DS4_SERVER_Q8_BATCH_RPB:-}"
export_pair_flag Q8_BATCH_SHARED_X "${DS4_SERVER_Q8_BATCH_SHARED_X:-0}"
export_pair_value Q8_BATCH_SHARED_X_BLOCKS "${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-}"
export_pair_value Q8_GROUPED_BATCH_TILE "${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-}"
export_pair_flag Q8_REPACK "${DS4_SERVER_Q8_REPACK:-0}"
export_pair_flag Q8_REPACK_SPLIT16 "${DS4_SERVER_Q8_REPACK_SPLIT16:-0}"
export_pair_flag Q8_WMMA_FAST "${DS4_SERVER_Q8_WMMA_FAST:-0}"
export_pair_flag Q8_HIPBLASLT "${DS4_SERVER_Q8_HIPBLASLT:-0}"
export_pair_value Q8_HIPBLASLT_MAX_TOKENS "${DS4_SERVER_Q8_HIPBLASLT_MAX_TOKENS:-}"
export_pair_flag MOE_EXPERT_BATCH "${DS4_SERVER_MOE_EXPERT_BATCH:-0}"
export_pair_value MOE_EXPERT_TILE "${DS4_SERVER_MOE_EXPERT_TILE:-}"
export_pair_value MOE_GATE_TILE "${DS4_SERVER_MOE_GATE_TILE:-}"
export_pair_value MOE_DOWN_TILE "${DS4_SERVER_MOE_DOWN_TILE:-}"
export_pair_value MOE_GATE_RPB "${DS4_SERVER_MOE_GATE_RPB:-}"
export_pair_value MOE_DOWN_RPB "${DS4_SERVER_MOE_DOWN_RPB:-}"
export_pair_flag MOE_EXPERT_SHARED_X "${DS4_SERVER_MOE_EXPERT_SHARED_X:-0}"
export_pair_flag MOE_EXPERT_SHARED_MID "${DS4_SERVER_MOE_EXPERT_SHARED_MID:-0}"
export_pair_flag MOE_Q8K_DOWN "${DS4_SERVER_MOE_Q8K_DOWN:-0}"
export_pair_value MOE_Q8K_DOWN_LAYERS "${DS4_SERVER_MOE_Q8K_DOWN_LAYERS:-}"
export_pair_flag MOE_Q8K_DOWN_DIRECT "${DS4_SERVER_MOE_Q8K_DOWN_DIRECT:-0}"
export_pair_value MOE_Q8K_DOWN_TILE "${DS4_SERVER_MOE_Q8K_DOWN_TILE:-}"
export_pair_flag MOE_WMMA_HOT "${DS4_SERVER_MOE_WMMA_HOT:-0}"
export_pair_value MOE_WMMA_GATE_HOT "${DS4_SERVER_MOE_WMMA_GATE_HOT:-}"
export_pair_value MOE_WMMA_DOWN_HOT "${DS4_SERVER_MOE_WMMA_DOWN_HOT:-}"
export_pair_value MOE_WMMA_LAYERS "${DS4_SERVER_MOE_WMMA_LAYERS:-}"
export_pair_value MOE_WMMA_MTILES "${DS4_SERVER_MOE_WMMA_MTILES:-}"
export_pair_flag MOE_WMMA_SPLIT_HOT "${DS4_SERVER_MOE_WMMA_SPLIT_HOT:-0}"
export_pair_flag MOE_WMMA_TILE_HOT "${DS4_SERVER_MOE_WMMA_TILE_HOT:-0}"
export_pair_flag MOE_WMMA_F16_MID "${DS4_SERVER_MOE_WMMA_F16_MID:-0}"
export_pair_flag MOE_WMMA_F16_MID_ALL "${DS4_SERVER_MOE_WMMA_F16_MID_ALL:-0}"
export_pair_flag MOE_WMMA_F16_DOWN "${DS4_SERVER_MOE_WMMA_F16_DOWN:-0}"
export_pair_flag MOE_WMMA_F16_DOWN_ALL "${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-0}"
export_pair_flag MOE_WMMA_DIRECT_SUM "${DS4_SERVER_MOE_WMMA_DIRECT_SUM:-0}"
export_pair_flag MOE_DENSE_HOT "${DS4_SERVER_MOE_DENSE_HOT:-0}"
export_pair_value MOE_DENSE_HOT_TOP "${DS4_SERVER_MOE_DENSE_HOT_TOP:-}"
export_pair_value MOE_DENSE_HOT_MIN "${DS4_SERVER_MOE_DENSE_HOT_MIN:-}"
export_pair_value MOE_DENSE_HOT_CACHE_MB "${DS4_SERVER_MOE_DENSE_HOT_CACHE_MB:-}"
export_pair_flag MOE_DENSE_HOT_NO_CACHE "${DS4_SERVER_MOE_DENSE_HOT_NO_CACHE:-0}"
export_pair_flag COPY_MODEL "${DS4_SERVER_COPY_MODEL:-0}"
export_pair_value COPY_MODEL_CHUNK_MB "${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}"
export_pair_flag CACHE_FINAL_Q8 "${DS4_SERVER_CACHE_FINAL_Q8:-0}"

if [[ -n "${DS4_SERVER_PERFLEVEL:-}" ]] && command -v rocm-smi >/dev/null 2>&1; then
  echo "ds4-bench-rocm-upstream: setting ROCm perflevel ${DS4_SERVER_PERFLEVEL}" >&2
  rocm-smi --setperflevel "${DS4_SERVER_PERFLEVEL}" >&2 || true
fi

if [[ "${DS4_BENCH_ALLOW_WITH_SERVER:-0}" != "1" ]]; then
  pidfile="${DS4_SERVER_PID:-/tmp/ds4-server.pid}"
  if [[ -f "$pidfile" ]]; then
    oldpid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ "$oldpid" =~ ^[0-9]+$ ]] && kill -0 "$oldpid" 2>/dev/null; then
      echo "ds4-bench-rocm-upstream: refusing to run while ds4-server pid $oldpid is alive" >&2
      echo "Set DS4_BENCH_ALLOW_WITH_SERVER=1 to override, or stop the server first." >&2
      exit 1
    fi
  fi
fi

make ds4-bench-rocm-upstream -j"$(nproc)"

MODEL="${DS4_MODEL:-$MODEL_DEFAULT}"
if [[ ! -f "$MODEL" ]]; then
  echo "ds4-bench-rocm-upstream: model not found: $MODEL" >&2
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  exec ./ds4-bench-rocm-upstream "$@"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${DS4_BENCH_CSV:-/tmp/ds4_bench_rocm_upstream_fast_full_ctx_${STAMP}.csv}"

exec ./ds4-bench-rocm-upstream \
  -m "$MODEL" \
  --prompt-file "${DS4_BENCH_PROMPT_FILE:-speed-bench/promessi_sposi.txt}" \
  --ctx-start "${DS4_BENCH_CTX_START:-2048}" \
  --ctx-max "${DS4_BENCH_CTX_MAX:-16384}" \
  --step-incr "${DS4_BENCH_STEP_INCR:-2048}" \
  --gen-tokens "${DS4_BENCH_GEN_TOKENS:-32}" \
  --csv "$CSV"
