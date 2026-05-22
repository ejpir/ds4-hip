#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DEFAULT="/home/nick/.cache/huggingface/hub/models--cyberneurova--CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/665c8e035e2602d12d28b84920808b158f337e09/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
STAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ds4_eval_rocm_upstream.sh [extra ds4-eval args]

Runs all embedded ds4-eval questions on the CyberNeurova GGUF using the
upstream-shaped ROCm eval binary and the quality-gated FAST_FULL ROCm
defaults used by production. Extra arguments are appended to ds4-eval, so you
can pass --questions N, -n N, --seed N, --prompt-final-newline, etc.

Defaults:
  model: CyberNeurova Q2_K GGUF
  binary: ./ds4-eval-rocm-upstream
  trace: /tmp/ds4-eval-cyber-rocm_TIMESTAMP.txt
  mode: --cuda --plain, all questions, default ds4-eval token budget (16000)
  ROCm flags: COPY_MODEL plus quality-gated FAST_FULL decode defaults only

Environment:
  DS4_MODEL=FILE                    Override model path
  DS4_EVAL_TRACE=FILE               Override trace path
  DS4_EVAL_BIN=FILE                 Override eval binary
  DS4_EVAL_TUI=1                    Use the split-screen TUI instead of --plain
  DS4_EVAL_SEED=N                   Add --seed N unless already passed
  DS4_EVAL_PROMPT_FINAL_NEWLINE=1   Append final prompt newline for parity checks
  DS4_EVAL_STOP_SERVER=1            Stop pidfile ds4-server before running
  DS4_EVAL_ALLOW_WITH_SERVER=1      Do not refuse if ds4-server is live
  DS4_LOCK_FILE=FILE                Override lock file

Examples:
  scripts/run_ds4_eval_rocm_upstream.sh
  scripts/run_ds4_eval_rocm_upstream.sh --questions 5 --seed 1
  DS4_EVAL_PROMPT_FINAL_NEWLINE=1 scripts/run_ds4_eval_rocm_upstream.sh --questions 5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="${DS4_MODEL:-$MODEL_DEFAULT}"
EVAL_BIN="${DS4_EVAL_BIN:-./ds4-eval-rocm-upstream}"
TRACE="${DS4_EVAL_TRACE:-/tmp/ds4-eval-cyber-rocm_${STAMP}.txt}"

if [[ ! -f "$MODEL" ]]; then
  echo "ds4-eval-rocm: model not found: $MODEL" >&2
  exit 2
fi

# A full eval copies/caches tens of GiB on the GPU. Avoid accidentally running
# it alongside the local API server unless explicitly requested.
if [[ -f /tmp/ds4-server.pid ]]; then
  oldpid="$(cat /tmp/ds4-server.pid 2>/dev/null || true)"
  if [[ "$oldpid" =~ ^[0-9]+$ ]] && kill -0 "$oldpid" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$oldpid/cmdline" 2>/dev/null | grep -q 'ds4-server'; then
      if [[ "${DS4_EVAL_STOP_SERVER:-0}" == "1" ]]; then
        echo "ds4-eval-rocm: stopping ds4-server pid $oldpid"
        kill "$oldpid" 2>/dev/null || true
        for _ in {1..60}; do
          kill -0 "$oldpid" 2>/dev/null || break
          sleep 0.5
        done
      elif [[ "${DS4_EVAL_ALLOW_WITH_SERVER:-0}" != "1" ]]; then
        echo "ds4-eval-rocm: ds4-server is live (pid=$oldpid)." >&2
        echo "Set DS4_EVAL_STOP_SERVER=1 to stop it, or DS4_EVAL_ALLOW_WITH_SERVER=1 to continue anyway." >&2
        exit 2
      fi
    fi
  fi
fi

make ds4-eval-rocm-upstream -j"$(nproc)"

# Optimized upstream-shaped ROCm settings. Keep these explicit rather than
# relying on DS4_SERVER_FAST_FULL translations from the server launcher.
export DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-eval-rocm-${STAMP}.lock}"
export DS4_CUDA_COPY_MODEL="${DS4_CUDA_COPY_MODEL:-1}"
export DS4_CUDA_Q8_PREQUANT_DECODE="${DS4_CUDA_Q8_PREQUANT_DECODE:-1}"
export DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW="${DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW:-1}"
export DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32="${DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32:-1}"

# Do not enable experimental opt-in kernels here.  Earlier eval parity runs
# used only the quality-gated FAST_FULL decode defaults above; broader WMMA,
# prefill-fast, and cuBLAS/Q8 batch toggles can change logits enough to alter
# greedy eval trajectories on aggressive Q2 models.

export DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-2048}"
export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$DS4_METAL_PREFILL_CHUNK}"

args=("$EVAL_BIN" -m "$MODEL" --cuda --trace "$TRACE")
if [[ "${DS4_EVAL_TUI:-0}" != "1" ]]; then
  args+=(--plain)
fi
if [[ -n "${DS4_EVAL_SEED:-}" ]]; then
  args+=(--seed "$DS4_EVAL_SEED")
fi
if [[ "${DS4_EVAL_PROMPT_FINAL_NEWLINE:-0}" == "1" ]]; then
  args+=(--prompt-final-newline)
fi
args+=("$@")

echo "ds4-eval-rocm: model=$MODEL"
echo "ds4-eval-rocm: trace=$TRACE"
echo "ds4-eval-rocm: binary=$EVAL_BIN"
echo "ds4-eval-rocm: command: ${args[*]}"

"${args[@]}"
