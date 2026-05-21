#!/usr/bin/env bash
set -euo pipefail

# Safe ds4-server launcher for local OpenAI-compatible testing.
# Defaults are conservative after the previous machine reset:
# - no rocm-smi perflevel change unless DS4_SERVER_PERFLEVEL or DS4_SERVER_FAST_FULL is set
# - managed HIP tensors unless DS4_SERVER_DEVICE_TENSORS=1
# - no optional final-Q8 cache/repack/top-only unless explicitly enabled

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/start_ds4_server.sh [-- extra ds4-server args]

Environment:
  DS4_MODEL=FILE                  GGUF model path
  DS4_SERVER_HOST=127.0.0.1       Bind host
  DS4_SERVER_PORT=8000            Bind port
  DS4_SERVER_CTX=4096             Context size, conservative default
  DS4_SERVER_TOKENS=1024          Default max output tokens
  DS4_SERVER_KV_DIR=/tmp/ds4-kv   Disk KV directory
  DS4_SERVER_KV_MB=2048           Disk KV budget MB
  DS4_SERVER_KV_ALIGN_TOKENS=512  Cold KV prefix alignment; helps <2k agent prompts reuse
  DS4_SERVER_LOG=/tmp/ds4-server.log
  DS4_SERVER_PID=/tmp/ds4-server.pid
  DS4_SERVER_TRACE=FILE           Optional trace file
  DS4_SERVER_FAST_FULL=1          Max-performance preset: high perflevel, device tensors, staged full-copy, best prefill+decode flags
  DS4_SERVER_PREFILL_HEARTBEAT_SEC=2  Prefill heartbeat interval; 0 disables
  DS4_SERVER_PREFILL_CHUNK=N      Set prefill chunk/allocation cap
  DS4_SERVER_DECODE_PREFILL=1     Safest prompt path: prefill via decode kernels
  DS4_SERVER_PREFILL_STAGE_PROFILE=1  Log/sync prefill stages to isolate crashes
  DS4_SERVER_PREFILL_RAW_FAST=1  Experimental FlashAttention-style raw SWA prefill kernel
  DS4_SERVER_PREFILL_MIXED_FAST=1  Experimental fast compressed/mixed prefill attention
  DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP=4  Fused group4 indexed-attention score/value path; 0 disables
  DS4_SERVER_ATTENTION_INDEXED_SPLIT_VALUE_GROUP=4  Scratch-backed grouped indexed-attention value pass fallback; 0 disables
  DS4_SERVER_ATTN_Q_B_CUBLAS=1 Use f16 GEMM/cache for the large q_b projection in upstream-shaped ROCm
  DS4_SERVER_ATTN_Q_B_PRELOAD=1 Preload q_b Q8_0 weights into the f16 cache
  DS4_SERVER_ATTN_Q_B_F16_OUT=1 Write q_b GEMM to f16 and fuse head norm/rope to float; FAST_FULL default 1
  DS4_SERVER_ATTENTION_OUTPUT_F16_OUT=1 Opt-in attention output f16 projection for large prefill; default min tokens 128
  DS4_SERVER_SHARED_DOWN_F16_OUT=1 Opt-in shared-down f16 projection for large prefill; default min tokens 128
  DS4_SERVER_Q8_BATCH_FAST=0|1   Batched Q8_0 prefill matmul is default-on in HIP; set 0 to disable
  DS4_SERVER_Q8_BATCH_TILE=32    Token tile for Q8 batch fast; passed through when set
  DS4_SERVER_Q8_BATCH_RPB=N      Rows/block for Q8 batch fast; default 32
  DS4_SERVER_Q8_BATCH_SHARED_X=0|1  LDS shared-X batched Q8 prefill is default-on; set 0 to disable
  DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS=32  K-block chunk for Q8 shared-X batch, 8|16|32
  DS4_SERVER_Q8_GROUPED_BATCH_TILE=32 Token tile for grouped attn_output_a Q8 batch, 2|4|8|16|32
  DS4_SERVER_Q8_REPACK=1         Opt-in eager q_b Q8_0 repack for decode; uses ~1.43 GiB VRAM
  DS4_SERVER_Q8_REPACK_SPLIT16=1 Opt-in split-major Q8_0 repack for attn_output/shared-down; uses ~3.2 GiB VRAM
  DS4_SERVER_Q8_WMMA_FAST=1      Opt-in q-side FP16 Q8 WMMA prefill path; uses ~3.35 GiB VRAM
                                Enabled by DS4_SERVER_FAST_FULL=1.
  DS4_SERVER_Q8_HIPBLASLT=1      Opt-in q-side hipBLASLt xsplit path for small prefill batches
  DS4_SERVER_Q8_HIPBLASLT_MAX_TOKENS=N  Max batch routed to hipBLASLt; default 256
  DS4_SERVER_MOE_EXPERT_BATCH=1  Experimental expert-bucketed Q2_K MoE for faster prefill
  DS4_SERVER_MOE_EXPERT_TILE=4|8|16  Legacy pair tile fallback for expert-bucketed Q2_K MoE; default 8
  DS4_SERVER_MOE_GATE_TILE=4|8|16  Gate/up pair tile; FAST_FULL default 4 in the scalar/shared-X path
  DS4_SERVER_MOE_DOWN_TILE=4|8|16  Down pair tile; FAST_FULL default 4 in the scalar/shared-mid path
  DS4_SERVER_MOE_GATE_RPB=N      Rows/block for expert-bucketed gate/up; e.g. 16 with shared-x experiment
  DS4_SERVER_MOE_DOWN_RPB=N      Rows/block for expert-bucketed down; e.g. 16 with shared-mid experiment
  DS4_SERVER_MOE_EXPERT_SHARED_X=1    Experimental LDS x tile reuse for gate/up; use with GATE_RPB>1
  DS4_SERVER_MOE_EXPERT_SHARED_MID=1  Experimental LDS mid tile reuse for down; use with DOWN_RPB>1
  DS4_SERVER_MOE_Q8K_DOWN=1      Opt-in Q8_K mid/down MoE path; exact-gated default layers are >=40
  DS4_SERVER_MOE_Q8K_DOWN_LAYERS=LIST Restrict Q8_K down to layers/ranges; overrides >=43 default
  DS4_SERVER_MOE_Q8K_DOWN_DIRECT=1    Use slower direct sum6 Q8_K-down variant instead of expert-batched
  DS4_SERVER_MOE_Q8K_DOWN_TILE=4|8|16 Expert-batched Q8_K-down pair tile; default 4
  DS4_SERVER_MOE_WMMA_HOT=1      Opt-in hot-bucket Q2_K WMMA MoE path. Enabled by DS4_SERVER_FAST_FULL=1.
  DS4_SERVER_MOE_WMMA_GATE_HOT=N Gate/up WMMA bucket threshold; FAST_FULL default 8
  DS4_SERVER_MOE_WMMA_DOWN_HOT=N Down WMMA bucket threshold; FAST_FULL default 8
  DS4_SERVER_MOE_WMMA_MTILES=4|8|16 Hot MoE WMMA token tiles/block; FAST_FULL default 16
  DS4_SERVER_MOE_WMMA_F16_DOWN_ALL=1 Store routed MoE down scratch as f16 in upstream-shaped ROCm; FAST_FULL default 1
  DS4_SERVER_MOE_WMMA_F16_SPLIT=1 Split low-count f16-hot MoE buckets to MTILES=4; FAST_FULL default 1
  DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN=64 Count cutoff for the f16-hot split path; FAST_FULL default 64
  DS4_SERVER_MOE_WMMA_X_F16=1 Preconvert MoE gate/up input activations to f16 once per chunk; FAST_FULL default 1
  DS4_SERVER_MOE_SLOT_PARTIAL=1 Route-slot partial down layout [slot, token, dim]; experimental/off by default
  DS4_SERVER_MOE_WMMA_LAYERS=LIST Restrict WMMA to layers/ranges, e.g. 14-42; diagnostic/experimental
  DS4_SERVER_COPY_MODEL=1        Copy GGUF tensor payload to HIP allocation using staged chunks
  DS4_SERVER_COPY_MODEL_CHUNK_MB=256  Staged full-copy chunk size
  DS4_SESSION_PROGRESS_RAW_MAX_TOKENS=512  Use cancellable layer-by-layer prefill above this
  DS4_SESSION_PROGRESS_CHUNK_TOKENS=512    Cap cancellable prefill chunks; 0 disables cap

Safety/perf toggles:
  DS4_SERVER_PERFLEVEL=high|auto  Optional rocm-smi --setperflevel
  DS4_SERVER_DEVICE_TENSORS=1     Use faster hipMalloc device tensors
  DS4_SERVER_TOP_ONLY=1           Greedy top-only decode
  DS4_SERVER_CACHE_FINAL_Q8=1     Cache final Q8 projection

Examples:
  scripts/start_ds4_server.sh
  DS4_SERVER_CTX=32768 scripts/start_ds4_server.sh
  DS4_SERVER_PERFLEVEL=high DS4_SERVER_DEVICE_TENSORS=1 scripts/start_ds4_server.sh
  scripts/start_ds4_server.sh -- --quality
EOF
  exit 0
fi

MODEL_DEFAULT="/home/nick/.cache/huggingface/hub/models--cyberneurova--CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/665c8e035e2602d12d28b84920808b158f337e09/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
MODEL="${DS4_MODEL:-$MODEL_DEFAULT}"
HOST="${DS4_SERVER_HOST:-127.0.0.1}"
PORT="${DS4_SERVER_PORT:-8000}"
CTX="${DS4_SERVER_CTX:-4096}"
TOKENS="${DS4_SERVER_TOKENS:-1024}"
KV_DIR="${DS4_SERVER_KV_DIR:-/tmp/ds4-kv}"
KV_MB="${DS4_SERVER_KV_MB:-2048}"
KV_ALIGN="${DS4_SERVER_KV_ALIGN_TOKENS:-512}"
LOG="${DS4_SERVER_LOG:-/tmp/ds4-server.log}"
PIDFILE="${DS4_SERVER_PID:-/tmp/ds4-server.pid}"
TRACE="${DS4_SERVER_TRACE:-}"

# One-command max-performance profile.  Individual env vars may still be set
# by the caller before this script to override these defaults.
if [[ "${DS4_SERVER_FAST_FULL:-0}" == "1" ]]; then
  export DS4_SERVER_PERFLEVEL="${DS4_SERVER_PERFLEVEL:-high}"
  export DS4_SERVER_PREFILL_CHUNK="${DS4_SERVER_PREFILL_CHUNK:-2048}"
  export DS4_SERVER_DEVICE_TENSORS="${DS4_SERVER_DEVICE_TENSORS:-1}"
  export DS4_SERVER_COPY_MODEL="${DS4_SERVER_COPY_MODEL:-1}"
  export DS4_SERVER_COPY_MODEL_CHUNK_MB="${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}"
  export DS4_SERVER_PREFILL_RAW_FAST="${DS4_SERVER_PREFILL_RAW_FAST:-1}"
  export DS4_SERVER_PREFILL_MIXED_FAST="${DS4_SERVER_PREFILL_MIXED_FAST:-1}"
  export DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP="${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-4}"
  export DS4_SERVER_ATTN_Q_B_CUBLAS="${DS4_SERVER_ATTN_Q_B_CUBLAS:-1}"
  export DS4_SERVER_ATTN_Q_B_PRELOAD="${DS4_SERVER_ATTN_Q_B_PRELOAD:-1}"
  export DS4_SERVER_ATTN_Q_B_F16_OUT="${DS4_SERVER_ATTN_Q_B_F16_OUT:-1}"
  # Keep f16-output projection shortcuts opt-in for serving/chat quality. The
  # backend also gates them to larger prefill chunks by *_MIN_TOKENS (default
  # 128) because short chat prompts were overly sensitive to half output.
  export DS4_SERVER_ATTENTION_OUTPUT_F16_OUT="${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT:-0}"
  export DS4_SERVER_SHARED_DOWN_F16_OUT="${DS4_SERVER_SHARED_DOWN_F16_OUT:-0}"
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
  export DS4_SERVER_MOE_WMMA_HOT="${DS4_SERVER_MOE_WMMA_HOT:-1}"
  export DS4_SERVER_MOE_WMMA_GATE_HOT="${DS4_SERVER_MOE_WMMA_GATE_HOT:-8}"
  export DS4_SERVER_MOE_WMMA_DOWN_HOT="${DS4_SERVER_MOE_WMMA_DOWN_HOT:-8}"
  export DS4_SERVER_MOE_WMMA_MTILES="${DS4_SERVER_MOE_WMMA_MTILES:-16}"
  export DS4_SERVER_MOE_WMMA_F16_DOWN_ALL="${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-1}"
  export DS4_SERVER_MOE_WMMA_F16_SPLIT="${DS4_SERVER_MOE_WMMA_F16_SPLIT:-1}"
  export DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN="${DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN:-64}"
  export DS4_SERVER_MOE_WMMA_X_F16="${DS4_SERVER_MOE_WMMA_X_F16:-1}"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ds4-server: model not found: $MODEL" >&2
  exit 1
fi

mkdir -p "$KV_DIR" "$(dirname "$LOG")" "$(dirname "$PIDFILE")"

# Build if missing/stale enough for normal make dependency tracking.
make ds4-server -j"$(nproc)"

# Stop previous server from our pidfile only. Avoid pkill -f self-match accidents.
if [[ -f "$PIDFILE" ]]; then
  oldpid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ "$oldpid" =~ ^[0-9]+$ ]] && kill -0 "$oldpid" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$oldpid/cmdline" 2>/dev/null | grep -q 'ds4-server'; then
      echo "ds4-server: stopping previous pid $oldpid"
      kill "$oldpid" 2>/dev/null || true
      for _ in {1..30}; do
        kill -0 "$oldpid" 2>/dev/null || break
        sleep 0.2
      done
      kill -0 "$oldpid" 2>/dev/null && kill -9 "$oldpid" 2>/dev/null || true
      lockpid="$(cat /tmp/ds4.lock 2>/dev/null || true)"
      if [[ "$lockpid" == "$oldpid" ]]; then
        rm -f /tmp/ds4.lock
      fi
    fi
  fi
  rm -f "$PIDFILE"
fi

# Clear stale ds4 single-process lock if the recorded process is gone.
if [[ -f /tmp/ds4.lock ]]; then
  lockpid="$(cat /tmp/ds4.lock 2>/dev/null || true)"
  if [[ "$lockpid" =~ ^[0-9]+$ ]] && ! kill -0 "$lockpid" 2>/dev/null; then
    echo "ds4-server: removing stale /tmp/ds4.lock for dead pid $lockpid"
    rm -f /tmp/ds4.lock
  fi
fi

# Optional ROCm perf level. Leave unset for safer/stable startup.
# Examples:
#   DS4_SERVER_PERFLEVEL=high scripts/start_ds4_server.sh
#   DS4_SERVER_PERFLEVEL=auto scripts/start_ds4_server.sh
if [[ -n "${DS4_SERVER_PERFLEVEL:-}" ]]; then
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "ds4-server: setting ROCm perflevel ${DS4_SERVER_PERFLEVEL}"
    rocm-smi --setperflevel "${DS4_SERVER_PERFLEVEL}" || true
  fi
fi

# Conservative default: managed tensors. For faster but less battle-tested mode:
#   DS4_SERVER_DEVICE_TENSORS=1 scripts/start_ds4_server.sh
if [[ "${DS4_SERVER_DEVICE_TENSORS:-0}" != "1" ]]; then
  export DS4_HIP_MANAGED_TENSORS=1
else
  unset DS4_HIP_MANAGED_TENSORS || true
fi

# Server progress/cancellation defaults. Long prompts use cancellable
# layer-by-layer prefill; cap chunks so a client abort is noticed sooner.
# Device tensors can drive much larger HIP power/driver bursts during batched
# prefill, so keep that mode on smaller chunks unless explicitly overridden.
if [[ -n "${DS4_SERVER_PREFILL_CHUNK:-}" ]]; then
  export DS4_METAL_PREFILL_CHUNK="$DS4_SERVER_PREFILL_CHUNK"
  export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$DS4_SERVER_PREFILL_CHUNK}"
elif [[ "${DS4_SERVER_DEVICE_TENSORS:-0}" == "1" ]]; then
  export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-128}"
  export DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-$DS4_SESSION_PROGRESS_CHUNK_TOKENS}"
else
  export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-512}"
fi

if [[ "${DS4_SERVER_DECODE_PREFILL:-0}" == "1" ]]; then
  export DS4_SESSION_DECODE_PREFILL=1
fi
if [[ "${DS4_SERVER_PREFILL_STAGE_PROFILE:-0}" == "1" ]]; then
  export DS4_METAL_LAYER_STAGE_PROFILE=1
  export DS4_METAL_Q_STAGE_PROFILE=1
fi
if [[ "${DS4_SERVER_PREFILL_RAW_FAST:-0}" == "1" ]]; then
  export DS4_HIP_PREFILL_RAW_FAST=1
fi
if [[ "${DS4_SERVER_PREFILL_MIXED_FAST:-0}" == "1" ]]; then
  export DS4_HIP_PREFILL_MIXED_FAST=1
fi
if [[ -n "${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-}" ]]; then
  export DS4_HIP_ATTENTION_INDEXED_FUSED_VALUE_GROUP="$DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP"
fi
if [[ -n "${DS4_SERVER_ATTENTION_INDEXED_SPLIT_VALUE_GROUP:-}" ]]; then
  export DS4_HIP_ATTENTION_INDEXED_SPLIT_VALUE_GROUP="$DS4_SERVER_ATTENTION_INDEXED_SPLIT_VALUE_GROUP"
fi
if [[ "${DS4_SERVER_ATTN_Q_B_CUBLAS:-0}" == "1" ]]; then
  export DS4_CUDA_ATTN_Q_B_CUBLAS=1
  export DS4_HIP_ATTN_Q_B_CUBLAS=1
fi
if [[ "${DS4_SERVER_ATTN_Q_B_PRELOAD:-0}" == "1" ]]; then
  export DS4_CUDA_ATTN_Q_B_PRELOAD=1
  export DS4_HIP_ATTN_Q_B_PRELOAD=1
fi
if [[ "${DS4_SERVER_ATTN_Q_B_F16_OUT:-0}" == "1" ]]; then
  export DS4_CUDA_ATTN_Q_B_F16_OUT=1
  export DS4_HIP_ATTN_Q_B_F16_OUT=1
fi
if [[ "${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT:-0}" == "1" ]]; then
  export DS4_CUDA_ATTENTION_OUTPUT_F16_OUT=1
  export DS4_HIP_ATTENTION_OUTPUT_F16_OUT=1
fi
if [[ "${DS4_SERVER_SHARED_DOWN_F16_OUT:-0}" == "1" ]]; then
  export DS4_CUDA_SHARED_DOWN_F16_OUT=1
  export DS4_HIP_SHARED_DOWN_F16_OUT=1
fi
if [[ "${DS4_SERVER_Q8_BATCH_FAST:-0}" == "1" ]]; then
  export DS4_HIP_Q8_BATCH_FAST=1
fi
if [[ -n "${DS4_SERVER_Q8_BATCH_TILE:-}" ]]; then
  export DS4_HIP_Q8_BATCH_TILE="$DS4_SERVER_Q8_BATCH_TILE"
fi
if [[ -n "${DS4_SERVER_Q8_BATCH_RPB:-}" ]]; then
  export DS4_HIP_Q8_BATCH_RPB="$DS4_SERVER_Q8_BATCH_RPB"
fi
if [[ "${DS4_SERVER_Q8_BATCH_SHARED_X:-0}" == "1" ]]; then
  export DS4_HIP_Q8_BATCH_FAST=1
  export DS4_HIP_Q8_BATCH_SHARED_X=1
fi
if [[ -n "${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-}" ]]; then
  export DS4_HIP_Q8_BATCH_SHARED_X_BLOCKS="$DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS"
fi
if [[ -n "${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-}" ]]; then
  export DS4_HIP_Q8_GROUPED_BATCH_TILE="$DS4_SERVER_Q8_GROUPED_BATCH_TILE"
fi
if [[ "${DS4_SERVER_Q8_REPACK:-0}" == "1" ]]; then
  export DS4_HIP_Q8_REPACK=1
fi
if [[ "${DS4_SERVER_Q8_REPACK_SPLIT16:-0}" == "1" ]]; then
  export DS4_HIP_Q8_REPACK_SPLIT16=1
fi
if [[ "${DS4_SERVER_Q8_WMMA_FAST:-0}" == "1" ]]; then
  export DS4_HIP_Q8_WMMA_FAST=1
fi
if [[ "${DS4_SERVER_Q8_HIPBLASLT:-0}" == "1" ]]; then
  export DS4_HIP_Q8_HIPBLASLT=1
fi
if [[ -n "${DS4_SERVER_Q8_HIPBLASLT_MAX_TOKENS:-}" ]]; then
  export DS4_HIP_Q8_HIPBLASLT_MAX_TOKENS="$DS4_SERVER_Q8_HIPBLASLT_MAX_TOKENS"
fi
if [[ "${DS4_SERVER_MOE_EXPERT_BATCH:-0}" == "1" ]]; then
  export DS4_HIP_MOE_EXPERT_BATCH=1
fi
if [[ -n "${DS4_SERVER_MOE_EXPERT_TILE:-}" ]]; then
  export DS4_HIP_MOE_EXPERT_TILE="$DS4_SERVER_MOE_EXPERT_TILE"
fi
if [[ -n "${DS4_SERVER_MOE_GATE_TILE:-}" ]]; then
  export DS4_HIP_MOE_GATE_TILE="$DS4_SERVER_MOE_GATE_TILE"
fi
if [[ -n "${DS4_SERVER_MOE_DOWN_TILE:-}" ]]; then
  export DS4_HIP_MOE_DOWN_TILE="$DS4_SERVER_MOE_DOWN_TILE"
fi
if [[ -n "${DS4_SERVER_MOE_GATE_RPB:-}" ]]; then
  export DS4_HIP_MOE_GATE_RPB="$DS4_SERVER_MOE_GATE_RPB"
fi
if [[ -n "${DS4_SERVER_MOE_DOWN_RPB:-}" ]]; then
  export DS4_HIP_MOE_DOWN_RPB="$DS4_SERVER_MOE_DOWN_RPB"
fi
if [[ "${DS4_SERVER_MOE_EXPERT_SHARED_X:-0}" == "1" ]]; then
  export DS4_HIP_MOE_EXPERT_SHARED_X=1
fi
if [[ "${DS4_SERVER_MOE_EXPERT_SHARED_MID:-0}" == "1" ]]; then
  export DS4_HIP_MOE_EXPERT_SHARED_MID=1
fi
if [[ "${DS4_SERVER_MOE_Q8K_DOWN:-0}" == "1" ]]; then
  export DS4_HIP_MOE_Q8K_DOWN=1
fi
if [[ -n "${DS4_SERVER_MOE_Q8K_DOWN_LAYERS:-}" ]]; then
  export DS4_HIP_MOE_Q8K_DOWN_LAYERS="$DS4_SERVER_MOE_Q8K_DOWN_LAYERS"
fi
if [[ "${DS4_SERVER_MOE_Q8K_DOWN_DIRECT:-0}" == "1" ]]; then
  export DS4_HIP_MOE_Q8K_DOWN_DIRECT=1
fi
if [[ -n "${DS4_SERVER_MOE_Q8K_DOWN_TILE:-}" ]]; then
  export DS4_HIP_MOE_Q8K_DOWN_TILE="$DS4_SERVER_MOE_Q8K_DOWN_TILE"
fi
if [[ "${DS4_SERVER_MOE_WMMA_HOT:-0}" == "1" ]]; then
  export DS4_HIP_MOE_WMMA_HOT=1
fi
if [[ -n "${DS4_SERVER_MOE_WMMA_GATE_HOT:-}" ]]; then
  export DS4_HIP_MOE_WMMA_GATE_HOT="$DS4_SERVER_MOE_WMMA_GATE_HOT"
fi
if [[ -n "${DS4_SERVER_MOE_WMMA_DOWN_HOT:-}" ]]; then
  export DS4_HIP_MOE_WMMA_DOWN_HOT="$DS4_SERVER_MOE_WMMA_DOWN_HOT"
fi
if [[ -n "${DS4_SERVER_MOE_WMMA_MTILES:-}" ]]; then
  export DS4_CUDA_MOE_WMMA_MTILES="$DS4_SERVER_MOE_WMMA_MTILES"
  export DS4_HIP_MOE_WMMA_MTILES="$DS4_SERVER_MOE_WMMA_MTILES"
fi
if [[ "${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-0}" == "1" ]]; then
  export DS4_CUDA_MOE_WMMA_F16_DOWN_ALL=1
  export DS4_HIP_MOE_WMMA_F16_DOWN_ALL=1
fi
if [[ "${DS4_SERVER_MOE_WMMA_F16_SPLIT:-0}" == "1" ]]; then
  export DS4_CUDA_MOE_WMMA_F16_SPLIT=1
  export DS4_HIP_MOE_WMMA_F16_SPLIT=1
fi
if [[ -n "${DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN:-}" ]]; then
  export DS4_CUDA_MOE_WMMA_F16_SPLIT_MIN="$DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN"
  export DS4_HIP_MOE_WMMA_F16_SPLIT_MIN="$DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN"
fi
if [[ "${DS4_SERVER_MOE_WMMA_X_F16:-0}" == "1" ]]; then
  export DS4_CUDA_MOE_WMMA_X_F16=1
  export DS4_HIP_MOE_WMMA_X_F16=1
fi
if [[ "${DS4_SERVER_MOE_SLOT_PARTIAL:-0}" == "1" ]]; then
  export DS4_CUDA_MOE_SLOT_PARTIAL=1
  export DS4_HIP_MOE_SLOT_PARTIAL=1
fi
if [[ -n "${DS4_SERVER_MOE_WMMA_LAYERS:-}" ]]; then
  export DS4_HIP_MOE_WMMA_LAYERS="$DS4_SERVER_MOE_WMMA_LAYERS"
fi
if [[ "${DS4_SERVER_COPY_MODEL:-0}" == "1" ]]; then
  export DS4_HIP_COPY_MODEL=1
fi
if [[ -n "${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}" ]]; then
  export DS4_HIP_COPY_MODEL_CHUNK_MB="$DS4_SERVER_COPY_MODEL_CHUNK_MB"
fi

# Optional speed knobs, off by default for stability.
if [[ "${DS4_SERVER_TOP_ONLY:-0}" == "1" ]]; then
  export DS4_METAL_TOP_ONLY_DECODE=1
fi
if [[ "${DS4_SERVER_CACHE_FINAL_Q8:-0}" == "1" ]]; then
  export DS4_HIP_CACHE_FINAL_Q8=1
fi

args=(
  ./ds4-server
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --ctx "$CTX"
  --tokens "$TOKENS"
  --kv-disk-dir "$KV_DIR"
  --kv-disk-space-mb "$KV_MB"
  --kv-cache-boundary-align-tokens "$KV_ALIGN"
)

if [[ -n "$TRACE" ]]; then
  args+=(--trace "$TRACE")
fi

# Pass additional server args after --, e.g.:
#   scripts/start_ds4_server.sh -- --quality
if [[ "${1:-}" == "--" ]]; then
  shift
fi
args+=("$@")

: > "$LOG"
echo "ds4-server: starting on http://$HOST:$PORT"
echo "ds4-server: log: $LOG"
echo "ds4-server: ctx=$CTX tokens=$TOKENS kv=$KV_DIR model=$MODEL"

nohup "${args[@]}" >> "$LOG" 2>&1 &
pid=$!
echo "$pid" > "$PIDFILE"

# Wait for process to stay alive and socket to appear.
for _ in {1..120}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ds4-server: failed to start; log follows:" >&2
    tail -120 "$LOG" >&2 || true
    exit 1
  fi
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(^|:)${PORT}$"; then
    echo "ds4-server: ready pid=$pid"
    tail -40 "$LOG" || true
    exit 0
  fi
  sleep 0.5
done

echo "ds4-server: process alive pid=$pid but port $PORT not observed yet; log follows:" >&2
tail -120 "$LOG" >&2 || true
exit 1
