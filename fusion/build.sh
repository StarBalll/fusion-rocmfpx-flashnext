#!/usr/bin/env bash
# =============================================================
# fusion-rocmfpx-flashnext 一键构建 (仓库内融合版)
#
# 布局: 本脚本位于仓库的 fusion/ 子目录, 仓库根 = llama.cpp 源码根
# 产物: /opt/llama-rocmfpx-next/bin/llama-server
# 幂等: 可重复执行; 已应用过的 cherry-pick 会自动跳过
#
# 用法:
#   ./fusion/build.sh                 # 全量: 校验补丁 → cmake → make → install
#   ./fusion/build.sh --skip-patches  # 跳过 upstream fetch / cherry-pick 校验
# =============================================================
set -euo pipefail

# ---------- 配置 ----------
FUSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$FUSION_DIR")"            # 仓库根 = ROCmFPX 源码
BUILD_DIR="$SRC_DIR/build"                    # cmake 构建目录 (已被 .gitignore 忽略)
INSTALL_DIR="${INSTALL_DIR:-/opt/llama-rocmfpx-next}"   # 安装前缀

ROCMFPX_REPO="https://github.com/ROCmFPX/ROCmFPX.git"
ROCMFPX_BRANCH="main"
ROCMFPX_REMOTE="upstream"                     # 仓库 origin 指向你的 GitHub fork, 上游用独立远端名
DRLUOTO_REMOTE="drluoto"
DRLUOTO_REPO="https://github.com/drluoto/llama.cpp.git"
DRLUOTO_BRANCH="strix-halo-flash-next"

# 按依赖顺序: gguf-py 张量注册 → 模型 MTP 头 → detached 草稿头加载器 → 长上下文修复 → GDN rsqrt
# ⚠️ 59a40b56f 必须要: 非共享 MTP 草稿 GGUF 也缺主干张量, 需要 detached-head 加载器
#    (crusaderky a82a58a) 才能加载; README 原分析"非共享头不需要它"有误
CHERRY_PICKS=(
  "d303eec923f92ccab7109e97d95cb5c1ab83e0d2"
  "d72620018f612bb05d16e585108e3440947a1f9f"
  "59a40b56ff79dff97a318a3c01ead70beb22a510"
  "5ad68897a6a53699a9a29cd4d4d809ca661a81b8"
  "590ac45bc19b8571beb52aa9de6cb1c8c0a52f3c"
)

SKIP_PATCHES=0
[ "${1:-}" = "--skip-patches" ] && SKIP_PATCHES=1

info() { echo -e "\033[1;36m▸ $*\033[0m"; }
ok()   { echo -e "\033[1;32m✓ $*\033[0m"; }
err()  { echo -e "\033[1;31m✗ $*\033[0m"; }

# ---------- 0. 源码根自检 ----------
if [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
  err "仓库根 $SRC_DIR 缺少 CMakeLists.txt — 请完整克隆本仓库 (fusion/ 只是仓库的一部分)"
  exit 1
fi

# ---------- 1. 同步上游并校验融合补丁 ----------
# 补丁是否已全部应用(幂等判断): cherry-pick 产生新 SHA, 只能按提交主题匹配
picks_applied() {
  local sha subj
  for sha in "${CHERRY_PICKS[@]}"; do
    subj="$(git -C "$SRC_DIR" log -1 --format=%s "$sha" 2>/dev/null)" || return 1
    git -C "$SRC_DIR" log --format=%s "$ROCMFPX_REMOTE/$ROCMFPX_BRANCH..HEAD" 2>/dev/null \
      | grep -qF "$subj" || return 1
  done
  return 0
}

if [ "$SKIP_PATCHES" = 1 ]; then
  info "跳过 upstream 同步 (--skip-patches)"
else
  if ! git -C "$SRC_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    err "$SRC_DIR 不是 git 仓库, 无法同步上游"; exit 1
  fi
  if ! git -C "$SRC_DIR" remote | grep -q "^$ROCMFPX_REMOTE$"; then
    git -C "$SRC_DIR" remote add "$ROCMFPX_REMOTE" "$ROCMFPX_REPO"
  fi
  info "fetch upstream $ROCMFPX_BRANCH ..."
  git -C "$SRC_DIR" fetch "$ROCMFPX_REMOTE" "$ROCMFPX_BRANCH"
  if ! git -C "$SRC_DIR" remote | grep -q "^$DRLUOTO_REMOTE$"; then
    git -C "$SRC_DIR" remote add "$DRLUOTO_REMOTE" "$DRLUOTO_REPO" 2>/dev/null || true
  fi
  git -C "$SRC_DIR" fetch "$DRLUOTO_REMOTE" "$DRLUOTO_BRANCH" 2>/dev/null || true

  # cherry-pick 需要提交身份 (容器里常未配置)
  git -C "$SRC_DIR" config user.name  >/dev/null 2>&1 || git -C "$SRC_DIR" config user.name  "fusion-build"
  git -C "$SRC_DIR" config user.email >/dev/null 2>&1 || git -C "$SRC_DIR" config user.email "fusion@localhost"

  if picks_applied; then
    ok "融合补丁已在 HEAD (幂等跳过)"
  else
    info "检测到上游更新, 重打 5 个融合提交 ..."
    git -C "$SRC_DIR" reset --hard "$ROCMFPX_REMOTE/$ROCMFPX_BRANCH"   # 干净基线(丢弃未完成补丁, 重打)
    cd "$SRC_DIR"
    for sha in "${CHERRY_PICKS[@]}"; do
      if git cat-file -e "$sha"^{commit} 2>/dev/null && git merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
        info "已包含: $(git log -1 --format=%h $sha) $(git log -1 --format=%s $sha | head -c60)"
        continue
      fi
      info "cherry-pick $sha ..."
      if ! git cherry-pick "$sha"; then
        err "cherry-pick $sha 冲突! 请手动解决后:"
        err "    git -C $SRC_DIR add -A && git -C $SRC_DIR cherry-pick --continue"
        err "    然后重新运行 ./fusion/build.sh"
        exit 1
      fi
    done
    info "⚠ 注意: reset 会回到上游基线+5补丁; prefill-opt 的 MMQ 调参提交"
    info "  (6b999d4 / f765ce1) 不在重打链里 — 若需要 prefill 优化, 请切回 prefill-opt 分支:"
    info "  git checkout prefill-opt && git rebase $ROCMFPX_REMOTE/$ROCMFPX_BRANCH"
  fi
fi
ok "ROCmFPX: $(git -C "$SRC_DIR" rev-parse --short HEAD) ($(git -C "$SRC_DIR" branch --show-current))"

# ---------- 2. 编译 ----------
# rocm.sh 引用 $LD_LIBRARY_PATH; set -u 下未定义时展开错误会无视 || true 直接杀掉整个脚本, 先兜底
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /etc/profile.d/rocm.sh 2>/dev/null || true
which hipcc >/dev/null 2>&1 || { err "hipcc 未安装 (需 ROCm HIP)"; exit 1; }

# 检查 hipcub 头 (drluoto MTP 相关; 若缺用 rocm-extras)
HIPCUB_HDR="/opt/rocm-extras/opt/rocm-7.2.4/include"
HIP_EXTRA=()
if [ ! -f /opt/rocm*/include/hipcub/hipcub.hpp ] && [ -d "$HIPCUB_HDR" ]; then
  HIP_EXTRA=(-DCMAKE_HIP_FLAGS="-isystem $HIPCUB_HDR")
  info "使用 rocm-extras 的 hipcub/rocprim 头"
fi

info "cmake 配置 (HIP, gfx1151, W4A4=OFF 铁律)..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$SRC_DIR" \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=OFF \
  -DGGML_HIP_ROCMI4_W4A4=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  "${HIP_EXTRA[@]}"

info "编译 (多核, 后台日志 $FUSION_DIR/build.log)..."
nproc >/dev/null && make -j"$(nproc)" > "$FUSION_DIR/build.log" 2>&1 || { tail -40 "$FUSION_DIR/build.log"; err "编译失败, 见 $FUSION_DIR/build.log"; exit 1; }

info "安装到 $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
make install > /dev/null 2>&1

ok "完成! 二进制: $INSTALL_DIR/bin/llama-server"
"$INSTALL_DIR/bin/llama-server" --version 2>&1 | grep -E "version|ROCmFPX" | head -2 || true
