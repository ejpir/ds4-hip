#!/usr/bin/env bash
# Build and optionally run the lightweight ROCm GPU API smoke test.
#
# By default this builds ds4_rocm.o if needed, links tools/rocm_api_smoke.c,
# checks ds4_gpu.h export parity, and runs the smoke binary.  It refuses to run
# while /tmp/ds4-server.pid is live unless --allow-server is supplied.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
CC_BIN="${CC:-cc}"
OUT="${TMPDIR:-/tmp}/rocm_api_smoke"
BUILD_ONLY=0
ALLOW_SERVER=0

usage() {
    cat <<EOF
usage: tools/run_rocm_api_smoke.sh [--build-only] [--allow-server] [--out PATH]

Options:
  --build-only     compile/link the smoke binary but do not run it
  --allow-server   run even if /tmp/ds4-server.pid is live
  --out PATH       output binary path (default: ${OUT})
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --allow-server)
            ALLOW_SERVER=1
            shift
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "--out requires a path" >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

make -C "$ROOT" ds4_rocm.o
python3 "$ROOT/tools/check_gpu_api_exports.py" --header "$ROOT/ds4_gpu.h" --backend "$ROOT/ds4_rocm.o"

OBJ="${OUT}.o"
"$CC_BIN" -O2 -I"$ROOT" -c "$ROOT/tools/rocm_api_smoke.c" -o "$OBJ"
"$HIPCC" "$OBJ" "$ROOT/ds4_rocm.o" \
    -lm -pthread -L/opt/rocm/lib -lhipblas -lhipblaslt \
    -o "$OUT"

echo "built $OUT"

if [[ "$BUILD_ONLY" -ne 0 ]]; then
    exit 0
fi

if [[ "$ALLOW_SERVER" -eq 0 && -f /tmp/ds4-server.pid ]]; then
    pid="$(tr -d '[:space:]' </tmp/ds4-server.pid || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "refusing to run ROCm API smoke while DS4 server is live: pid=$pid" >&2
        echo "stop it first or pass --allow-server if the GPU is isolated" >&2
        exit 2
    fi
fi

"$OUT"
