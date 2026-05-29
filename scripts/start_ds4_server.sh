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
Usage: scripts/start_ds4_server.sh [--original] [-- extra ds4-server args]

Options:
  --original                      Use the original Antirez GGUF shipped/downloaded under ./gguf
                                  instead of the local abliterated default.

Environment:
  DS4_MODEL=FILE                  GGUF model path; overrides --original/default selection
  DS4_SERVER_HOST=127.0.0.1       Bind host
  DS4_SERVER_PORT=8000            Bind port
  DS4_SERVER_CTX=131072           Context size, standard default (128k)
  DS4_SERVER_TOKENS=8192          Default max output tokens
  DS4_SERVER_KV_DIR=/tmp/ds4-kv   Disk KV directory
  DS4_SERVER_KV_MB=2048           Disk KV budget MB
  DS4_SERVER_KV_ALIGN_TOKENS=512  Cold KV prefix alignment; helps <2k agent prompts reuse
  DS4_SERVER_LOG=/tmp/ds4-server.log
  DS4_SERVER_PID=/tmp/ds4-server.pid
  DS4_SERVER_TRACE=FILE           Optional trace file
  DS4_SERVER_FAST_FULL=1          Max-performance preset: high perflevel, device tensors, dynamic prefill, staged full-copy, best prefill+decode flags
  DS4_SERVER_PREFILL_HEARTBEAT_SEC=2  Prefill heartbeat interval; 0 disables
  DS4_SERVER_PREFILL_CHUNK=N      Set prefill chunk/allocation cap
  DS4_SERVER_DYNAMIC_PREFILL=1    Per-request prefill chunk cap from new suffix size; FAST_FULL default 1
  DS4_SERVER_DYNAMIC_PREFILL_MIN/MAX=N  Clamp dynamic chunk cap; defaults 128/4096
  DS4_SERVER_DECODE_PREFILL=1     Safest prompt path: prefill via decode kernels
  DS4_SERVER_PREFILL_STAGE_PROFILE=1  Log/sync prefill stages to isolate crashes
  DS4_SERVER_PREFILL_RAW_FAST=1  Experimental FlashAttention-style raw SWA prefill kernel
  DS4_SERVER_PREFILL_MIXED_FAST=1  Experimental fast compressed/mixed prefill attention
  DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP=4  Fused group4 indexed-attention score/value path; 0 disables
  DS4_SERVER_ATTENTION_INDEXED_SPLIT_VALUE_GROUP=4  Scratch-backed grouped indexed-attention value pass fallback; 0 disables
  DS4_SERVER_INDEXED_HEADS32=1   Upstream-shaped ROCm indexed-attention 32-head block; FAST_FULL default 1
  DS4_SERVER_ATTN_Q_B_CUBLAS=1 Use f16 GEMM/cache for the large q_b projection in upstream-shaped ROCm
  DS4_SERVER_ATTN_Q_B_PRELOAD=1 Preload q_b Q8_0 weights into the f16 cache
  DS4_SERVER_ATTN_Q_B_F16_OUT=1 Write q_b GEMM to f16 and fuse head norm/rope to float; FAST_FULL default 1
  DS4_SERVER_ATTENTION_OUTPUT_F16_OUT=0 Disable FAST_FULL default attention output f16 projection
  DS4_SERVER_ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS=N Optional attention f16-output minimum; default 0
  DS4_SERVER_SHARED_DOWN_F16_OUT=0 Disable FAST_FULL default shared-down f16 projection
  DS4_SERVER_SHARED_DOWN_F16_OUT_MIN_TOKENS=N Optional shared-down f16-output minimum; default 0
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
  DS4_SERVER_Q8_WMMA_4W=1        Experimental 4-wave 64x64 Q8 WMMA path for large prefill batches; FAST_FULL default 1
  DS4_SERVER_Q8_WMMA_4W_MIN_TOKENS=N Minimum token batch for 4-wave Q8 WMMA; default 256
  DS4_SERVER_Q8_WMMA_4W_MIN_OUT=N Minimum output rows for 4-wave Q8 WMMA; default 1024
  DS4_SERVER_Q8_PREQUANT_DECODE=0 Disable FAST_FULL q-side prequantized Q8 decode matvecs in upstream-shaped ROCm
  DS4_SERVER_Q8_DECODE_RPB=1|2|4|8|16|32 Rows/block for single-token Q8 decode matvecs; FAST_FULL default 1
  DS4_SERVER_Q8_HC_DECODE_RPB=1|2|4|8|16|32 Rows/block for fused Q8 HC-expand decode matvecs; FAST_FULL default 16
  DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB=1|2|4|8|16|32 Rows/block for decode attn_output_a low projection; FAST_FULL default 32
  DS4_SERVER_OLDHIP_ATTENTION_DECODE=1 Use the old-HIP decode attention kernel; FAST_FULL default 1
  DS4_SERVER_QKV_PAIR_DECODE=1      Opt in shared-quant Q/KV decode projection pair; FAST_FULL default 1
  DS4_SERVER_OVERLAP_SHARED_GATE_UP=0 Disable FAST_FULL async shared-expert gate/up overlap
  DS4_SERVER_ATTN_OUT_LOW_SPLITK=1 Restore old split-K decode attn_output_a path; FAST_FULL defaults to prequant
  DS4_SERVER_SHARED_GATE_UP_FUSED_W32=1 Restore old fused shared gate/up float-row decode path; FAST_FULL defaults to prequant pair
  DS4_SERVER_Q8_HIPBLASLT=1      Opt-in q-side hipBLASLt xsplit path for small prefill batches
  DS4_SERVER_Q8_HIPBLASLT_MAX_TOKENS=N  Max batch routed to hipBLASLt; default 256
  DS4_SERVER_MOE_EXPERT_BATCH=1  Experimental expert-bucketed Q2_K MoE for faster prefill
  DS4_SERVER_MOE_EXPERT_TILE=4|8|16  Legacy pair tile fallback for expert-bucketed Q2_K MoE; default 8
  DS4_SERVER_MOE_GATE_TILE=4|8|16  Gate/up pair tile; FAST_FULL default 4 in the scalar/shared-X path
  DS4_SERVER_MOE_DOWN_TILE=4|8|16  Down pair tile; FAST_FULL default 4 in the scalar/shared-mid path
  DS4_SERVER_MOE_GATE_RPB=N      Rows/block for expert-bucketed gate/up; e.g. 16 with shared-x experiment
  DS4_SERVER_MOE_DOWN_RPB=N      Rows/block for expert-bucketed down; e.g. 16 with shared-mid experiment
  DS4_SERVER_MOE_DECODE_RPB=N    Rows/block for exact single-token Q2_K MoE decode; FAST_FULL default 1
  DS4_SERVER_MOE_EXPERT_SHARED_X=1    Experimental LDS x tile reuse for gate/up; use with GATE_RPB>1
  DS4_SERVER_MOE_EXPERT_SHARED_MID=1  Experimental LDS mid tile reuse for down; use with DOWN_RPB>1
  DS4_SERVER_MOE_DECODE_Q8K_DOWN=1 Opt-in Q8_K mid/down MoE decode path; FAST_FULL default 1
  DS4_SERVER_MOE_Q8K_DOWN=1      Legacy alias for Q8_K mid/down MoE decode path
  DS4_SERVER_MOE_Q8K_DOWN_LAYERS=LIST Legacy/no-op decode restriction knob
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
  DS4_SERVER_COMPACT_TOOL_CONTEXT=1 Compact duplicate tool results in rendered prompt context
  DS4_SERVER_TOOL_CONTEXT_COMPACT_UNIQUE=1 Also compact older unique tool results; off by default
  DS4_SERVER_TOOL_CONTEXT_KEEP_LAST=8 Keep this many latest unique tool results full when compacting unique outputs
  DS4_SERVER_TOOL_CONTEXT_OLDER_CHARS=1600 Older unique tool-result excerpt bytes/chars when COMPACT_UNIQUE=1
  DS4_SERVER_TOOL_CONTEXT_DUP_CHARS=280 Duplicate tool-result excerpt bytes/chars

Safety/perf toggles:
  DS4_SPEC_DRAFTER=ngram          Opt-in prompt-lookup/n-gram speculative drafter for greedy decode
  DS4_SPEC_MAX_DRAFT=16           Maximum n-gram draft length; verifier accepts full chunks only
  DS4_SPEC_NGRAM_REJECT_COOLDOWN=128  Tokens to skip after a failed n-gram verification
  DS4_SERVER_PERFLEVEL=high|auto  Optional rocm-smi --setperflevel; set empty/off/skip to bypass
  DS4_SERVER_DEVICE_TENSORS=1     Use faster hipMalloc device tensors
  DS4_SERVER_TOP_ONLY=1           Greedy top-only decode
  DS4_SERVER_CACHE_FINAL_Q8=1     Cache final Q8 projection

Examples:
  scripts/start_ds4_server.sh
  scripts/start_ds4_server.sh --original
  DS4_SERVER_FAST_FULL=1 scripts/start_ds4_server.sh --original
  DS4_SERVER_CTX=65536 scripts/start_ds4_server.sh
  DS4_SERVER_PERFLEVEL=high DS4_SERVER_DEVICE_TENSORS=1 scripts/start_ds4_server.sh
  scripts/start_ds4_server.sh -- --quality
EOF
  exit 0
fi

USE_ORIGINAL=0
SERVER_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --original)
      USE_ORIGINAL=1
      shift
      ;;
    --)
      shift
      SERVER_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      SERVER_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ABLITERATED_MODEL_DEFAULT="/home/nick/.cache/huggingface/hub/models--cyberneurova--CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/665c8e035e2602d12d28b84920808b158f337e09/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
ORIGINAL_MODEL_DEFAULT="$ROOT_DIR/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
if [[ "$USE_ORIGINAL" == "1" ]]; then
  MODEL_DEFAULT="$ORIGINAL_MODEL_DEFAULT"
else
  MODEL_DEFAULT="$ABLITERATED_MODEL_DEFAULT"
fi
MODEL="${DS4_MODEL:-$MODEL_DEFAULT}"
HOST="${DS4_SERVER_HOST:-127.0.0.1}"
PORT="${DS4_SERVER_PORT:-8000}"
CTX="${DS4_SERVER_CTX:-131072}"
TOKENS="${DS4_SERVER_TOKENS:-8192}"
KV_DIR="${DS4_SERVER_KV_DIR:-/tmp/ds4-kv}"
KV_MB="${DS4_SERVER_KV_MB:-4096}"
KV_ALIGN="${DS4_SERVER_KV_ALIGN_TOKENS:-512}"
LOG="${DS4_SERVER_LOG:-/tmp/ds4-server.log}"
PIDFILE="${DS4_SERVER_PID:-/tmp/ds4-server.pid}"
TRACE="${DS4_SERVER_TRACE:-}"
export DS4_SERVER_COMPACT_TOOL_CONTEXT="${DS4_SERVER_COMPACT_TOOL_CONTEXT:-1}"
export DS4_SERVER_TOOL_CONTEXT_COMPACT_UNIQUE="${DS4_SERVER_TOOL_CONTEXT_COMPACT_UNIQUE:-0}"
export DS4_SERVER_TOOL_CONTEXT_KEEP_LAST="${DS4_SERVER_TOOL_CONTEXT_KEEP_LAST:-8}"
export DS4_SERVER_TOOL_CONTEXT_OLDER_CHARS="${DS4_SERVER_TOOL_CONTEXT_OLDER_CHARS:-1600}"
export DS4_SERVER_TOOL_CONTEXT_DUP_CHARS="${DS4_SERVER_TOOL_CONTEXT_DUP_CHARS:-280}"
SERVER_BIN="./ds4-server-rocm-upstream"
SERVER_MAKE_TARGET="ds4-server-rocm-upstream"

source "$ROOT_DIR/scripts/rocm_settings.sh"

# One-command max-performance profile. Individual env vars may still be set
# by the caller before this script to override these defaults. Shared ROCm
# defaults live in scripts/rocm_settings.sh.
if [[ "${DS4_SERVER_FAST_FULL:-0}" == "1" ]]; then
  ds4_rocm_apply_fast_full_defaults 4096
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ds4-server: model not found: $MODEL" >&2
  exit 1
fi

mkdir -p "$KV_DIR" "$(dirname "$LOG")" "$(dirname "$PIDFILE")"

# Build if missing/stale enough for normal make dependency tracking.
make "$SERVER_MAKE_TARGET" -j"$(nproc)"

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
if [[ "${DS4_SERVER_PERFLEVEL:-}" == "off" || "${DS4_SERVER_PERFLEVEL:-}" == "skip" || "${DS4_SERVER_PERFLEVEL:-}" == "none" ]]; then
  DS4_SERVER_PERFLEVEL=""
fi
if [[ -n "${DS4_SERVER_PERFLEVEL:-}" ]]; then
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "ds4-server: setting ROCm perflevel ${DS4_SERVER_PERFLEVEL}"
    rocm-smi --setperflevel "${DS4_SERVER_PERFLEVEL}" || true
  fi
fi

# Tensor placement and prefill chunk env are shared with the other ROCm scripts.
ds4_rocm_apply_tensor_env
ds4_rocm_apply_prefill_env 128 512

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

# Upstream-shaped ROCm consumes the historical CUDA/HIP env names. The shared
# helper translates DS4_SERVER_* knobs to both alias families.
ds4_rocm_export_runtime_env

# Optional speed knobs, off by default for stability.
if [[ "${DS4_SERVER_TOP_ONLY:-0}" == "1" ]]; then
  export DS4_METAL_TOP_ONLY_DECODE=1
fi
if [[ "${DS4_SERVER_CACHE_FINAL_Q8:-0}" == "1" ]]; then
  export DS4_HIP_CACHE_FINAL_Q8=1
fi

args=(
  "$SERVER_BIN"
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
args+=("${SERVER_EXTRA_ARGS[@]}")

: > "$LOG"
echo "ds4-server: starting $SERVER_BIN on http://$HOST:$PORT"
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
