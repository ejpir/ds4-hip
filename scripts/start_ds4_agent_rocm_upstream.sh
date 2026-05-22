#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DEFAULT="/home/nick/.cache/huggingface/hub/models--cyberneurova--CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF/snapshots/665c8e035e2602d12d28b84920808b158f337e09/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"

usage() {
  cat <<'EOF'
Usage:
  scripts/start_ds4_agent_rocm_upstream.sh [ds4-agent args]

Starts the upstream-shaped ROCm ds4-agent with the CyberNeurova Q2_K GGUF and
quality-gated ROCm FAST_FULL decode defaults. Extra arguments are passed through
to ds4-agent, so you can add -p TEXT, --nothink, --trace FILE, etc.

Defaults:
  DS4_MODEL=$CyberNeurova_Q2_K_GGUF
  DS4_AGENT_CTX=131072
  DS4_AGENT_TOKENS=8192
  DS4_AGENT_PERFLEVEL=high
  backend=--cuda

Safety:
  By default this refuses to run while the production ds4-server pidfile process
  is alive. Set DS4_AGENT_STOP_SERVER=1 to stop that server first, or
  DS4_AGENT_ALLOW_WITH_SERVER=1 to bypass the check.

Useful environment overrides:
  DS4_MODEL=FILE
  DS4_AGENT_CTX=N
  DS4_AGENT_TOKENS=N
  DS4_AGENT_TRACE=FILE
  DS4_AGENT_STOP_SERVER=1
  DS4_AGENT_ALLOW_WITH_SERVER=1
  DS4_AGENT_PERFLEVEL=auto    Or empty/off/skip/none to bypass rocm-smi changes

Examples:
  scripts/start_ds4_agent_rocm_upstream.sh
  scripts/start_ds4_agent_rocm_upstream.sh -p "Inspect this repo."
  DS4_AGENT_TOKENS=16000 scripts/start_ds4_agent_rocm_upstream.sh --think
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

backend_present() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --cuda|--hip|--metal|--cpu|--backend|--backend=*) return 0 ;;
    esac
  done
  return 1
}

pidfile="${DS4_SERVER_PID:-/tmp/ds4-server.pid}"
if [[ -f "$pidfile" ]]; then
  oldpid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ "$oldpid" =~ ^[0-9]+$ ]] && kill -0 "$oldpid" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$oldpid/cmdline" 2>/dev/null | grep -q 'ds4-server'; then
      if [[ "${DS4_AGENT_STOP_SERVER:-0}" == "1" ]]; then
        echo "ds4-agent-rocm-upstream: stopping ds4-server pid $oldpid" >&2
        kill "$oldpid" 2>/dev/null || true
        for _ in {1..60}; do
          kill -0 "$oldpid" 2>/dev/null || break
          sleep 0.5
        done
        kill -0 "$oldpid" 2>/dev/null && kill -9 "$oldpid" 2>/dev/null || true
        rm -f "$pidfile" /tmp/ds4.lock
      elif [[ "${DS4_AGENT_ALLOW_WITH_SERVER:-0}" != "1" ]]; then
        echo "ds4-agent-rocm-upstream: refusing to run while ds4-server pid $oldpid is alive" >&2
        echo "Set DS4_AGENT_STOP_SERVER=1 to stop it, or DS4_AGENT_ALLOW_WITH_SERVER=1 to bypass." >&2
        exit 1
      fi
    fi
  fi
fi

if [[ -f /tmp/ds4.lock ]]; then
  lockpid="$(cat /tmp/ds4.lock 2>/dev/null || true)"
  if [[ "$lockpid" =~ ^[0-9]+$ ]] && ! kill -0 "$lockpid" 2>/dev/null; then
    echo "ds4-agent-rocm-upstream: removing stale /tmp/ds4.lock for dead pid $lockpid" >&2
    rm -f /tmp/ds4.lock
  fi
fi

DS4_AGENT_PERFLEVEL="${DS4_AGENT_PERFLEVEL:-high}"
if [[ "$DS4_AGENT_PERFLEVEL" == "off" || "$DS4_AGENT_PERFLEVEL" == "skip" || "$DS4_AGENT_PERFLEVEL" == "none" ]]; then
  DS4_AGENT_PERFLEVEL=""
fi
if [[ -n "$DS4_AGENT_PERFLEVEL" ]] && command -v rocm-smi >/dev/null 2>&1; then
  echo "ds4-agent-rocm-upstream: setting ROCm perflevel $DS4_AGENT_PERFLEVEL" >&2
  rocm-smi --setperflevel "$DS4_AGENT_PERFLEVEL" >&2 || true
fi

make ds4-agent-rocm-upstream -j"$(nproc)"

MODEL="${DS4_MODEL:-$MODEL_DEFAULT}"
if [[ ! -f "$MODEL" ]]; then
  echo "ds4-agent-rocm-upstream: model not found: $MODEL" >&2
  exit 1
fi

# Quality-gated production decode defaults. Keep experimental opt-in kernels out
# of the agent launcher unless the caller explicitly exports them.
export DS4_CUDA_COPY_MODEL="${DS4_CUDA_COPY_MODEL:-1}"
export DS4_CUDA_Q8_PREQUANT_DECODE="${DS4_CUDA_Q8_PREQUANT_DECODE:-1}"
export DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW="${DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW:-1}"
export DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32="${DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32:-1}"
export DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-2048}"
export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$DS4_METAL_PREFILL_CHUNK}"

args=("$@")
if ! arg_present -m --model "${args[@]}"; then
  args=(-m "$MODEL" "${args[@]}")
fi
if ! arg_present -c --ctx "${args[@]}"; then
  args=(--ctx "${DS4_AGENT_CTX:-131072}" "${args[@]}")
fi
if ! arg_present -n --tokens "${args[@]}"; then
  args=(--tokens "${DS4_AGENT_TOKENS:-8192}" "${args[@]}")
fi
if ! backend_present "${args[@]}"; then
  args=(--cuda "${args[@]}")
fi
if [[ -n "${DS4_AGENT_TRACE:-}" ]] && ! arg_present "" --trace "${args[@]}"; then
  args=(--trace "$DS4_AGENT_TRACE" "${args[@]}")
fi

echo "ds4-agent-rocm-upstream: model=$MODEL" >&2
echo "ds4-agent-rocm-upstream: ctx=${DS4_AGENT_CTX:-131072} tokens=${DS4_AGENT_TOKENS:-8192}" >&2
exec ./ds4-agent-rocm-upstream "${args[@]}"
