#!/usr/bin/env bash
# =============================================================
# 启动融合构建的 Flash-Next 测试服务器
# 约定: 测试端口 8132, 不与已注册模型冲突 (qwen38fn2=8131)
# 输出重定向到 /tmp/llama-fusion.out (TTY 竞态保护)
# =============================================================
set -euo pipefail

BIN="/opt/llama-rocmfpx-next/bin/llama-server"
MODEL="/data/models/qwen38-flash-next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
MTP="/data/models/qwen38-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
PORT=8132
ALIAS="qwen38-flash-next-fusion"
LOG="/tmp/llama-fusion.out"

if [ ! -x "$BIN" ]; then
  echo "✗ 二进制不存在: $BIN"
  echo "   先运行 ./build.sh"
  exit 1
fi

# 如果已有进程占用了端口, 停掉
if pgrep -f "port $PORT" >/dev/null 2>&1; then
  echo "⚠ 端口 $PORT 已被占用, 先停掉..."
  pkill -f "port $PORT" 2>/dev/null || true
  sleep 3
fi

echo "═══ 启动融合构建: $ALIAS ═══"
echo "  端口: $PORT  | 日志: $LOG"

# ⚠️ 自定义 activate 末尾会 source rocm.sh (其引用 $LD_LIBRARY_PATH);
#    set -u 下未绑定时展开错误会无视 || true 直接杀死整个脚本 → 兜底必须在 source activate 之前
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PS1="${PS1:-}"
export VIRTUAL_ENV_DISABLE_PROMPT=1
source /root/llama-env/bin/activate
source /etc/profile.d/rocm.sh 2>/dev/null || true
# 融合构建为动态链接, 库在 install 前缀的 lib/ 下 (rocm.sh 会重置 LD_LIBRARY_PATH, 须在其后追加)
export LD_LIBRARY_PATH="/opt/llama-rocmfpx-next/lib:${LD_LIBRARY_PATH:-}"
export GGML_CUDA_DISABLE_GRAPHS=1
export LLAMA_HIP_ENABLE_COMPOSABLE_KERNEL=0

# 注意: --no-mmap 与 --load-mode dio 同时给出时后者优先; 无冲突
"$BIN" \
  -m "$MODEL" \
  --spec-draft-model "$MTP" \
  -ngl 99 --flash-attn on --no-mmap \
  --ctx-size 32768 \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 6 --spec-draft-p-min 0.7 --spec-draft-ngl 99 \
  --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24 \
  --jinja --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
  --host 127.0.0.1 --port "$PORT" --alias "$ALIAS" &>"$LOG"

echo "⚠ 服务器已退出 (日志: $LOG)"