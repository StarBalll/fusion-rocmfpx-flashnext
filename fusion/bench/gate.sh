# 35B Q6_K pp512 regression gate
# Compares llama-bench output against the recorded acceptance run (bench/data/final-ab).
# Usage: run on the target machine after ./fusion/build.sh
#   ./fusion/bench/gate.sh <path-to-Q6_K-gguf> [min-pp512]
set -euo pipefail

MODEL="${1:?usage: gate.sh <model.gguf> [min-pp512]}"
MIN_PP="${2:-1000}"
BIN_DIR="${INSTALL_DIR:-/opt/llama-rocmfpx-next}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}/opt/llama-rocmfpx-next/lib:$BIN_DIR/lib"
BENCH="$BIN_DIR/bin/llama-bench"
[ -x "$BENCH" ] || { echo "llama-bench not found at $BENCH (build first)"; exit 1; }

echo "== 35B Q6_K pp512 gate: >= ${MIN_PP} t/s =="
OUT=$("$BENCH" -m "$MODEL" -p 512 -n 0 -ngl 99 -fa on -r 3 2>/dev/null || \
      "$BENCH" -m "$MODEL" -p 512 -n 0 -ngl 99 -fa 1 -r 3 2>/dev/null)
echo "$OUT" | grep -E '^\|' || { echo "no bench output"; exit 1; }

PP=$(echo "$OUT" | grep -oE '\|\s+[0-9]+\.[0-9]+ ±' | tail -1 | grep -oE '[0-9]+\.[0-9]+')
[ -n "$PP" ] || { echo "could not parse pp512 value"; exit 1; }

awk -v v="$PP" -v m="$MIN_PP" 'BEGIN {
  if (v+0 >= m+0) { printf "PASS: pp512 = %s t/s (>= %s)\n", v, m; exit 0 }
  printf "FAIL: pp512 = %s t/s (< %s)\n", v, m; exit 1 }'
