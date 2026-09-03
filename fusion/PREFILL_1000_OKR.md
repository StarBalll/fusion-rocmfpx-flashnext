# PREFILL_1000_OKR.md — 为融合构建找回旧 rocmfpx 的 prefill 优化

## 目标

在融合构建（`fusion-rocmfpx-flashnext`）上，**35B Q6_K 的 pp512 prefill 达到 1000+ t/s**（当前 787，旧 rocmfpx 为 1101）。

## 背景：为什么速度丢了

| 构建 | 35B Q6_K pp512 | 来源 |
|---|---|---|
| 旧 rocmfpx (c49ebdb) | **1101 t/s** | charlie12345 的 ROCmFPX 分支，有自定义 RDNA3.5 kernel 调参 |
| 新 ROCmFPX main (融合基础) | 787 t/s | 追踪上游 master，但遗失旧定制调参 |
| stock b10752 | 794 t/s | 上游标准，无特调 |

实测证实：新 ROCmFPX/ROCmFPX 仓库的 README 写明了"legacy lineage 不会替换当前实现"——旧的自定义 kernel 优化没有被带进新版。

## 首要嫌疑人：MMQ workgroup geometry（e391458）

### 旧 rocmfpx 的改动

单个提交 `e391458`（"hip: tune RDNA3.5 MMQ workgroup geometry"），只改了一个文件：

**`ggml/src/ggml-cuda/mmq.cuh`**

```
// 旧版（含 e391458 调参）:
// gfx1151 (RDNA3.5): mmq_y = 64, nwarps = 4
// 注释: "64 rows and 4 warps reduce LDS staging stalls compared with 128/8"

// 新版上游（遗失调参）:
// gfx1151: 使用通用默认值 mmq_y = 128, nwarps = 8
```

### 需要改的代码位置（新版上游已重构，函数名可能变）

**旧版函数**（在 `mmq.cuh` 中）：
- `get_mmq_y(cc)` → 对 RDNA3_5 返回 64
- `get_mmq_nwarps(cc)` → 对 RDNA3_5 返回 4

**新版上游对应位置**：在 `src/ggml/src/ggml-cuda/mmq.cuh` 中找 `RDNA3_5` 或 `get_mmq_y` / `get_mmq_nwarps` 或等价的 `I` 模板参数定义。新上游可能已将 `mmq_y` 替换为模板参数 `I`，但逻辑相同：对 gfx1151 用 64 行/4 warp。

### 预期原理

MMQ（混合精度矩阵量化）在 prefill 阶段执行大批量 GEMM。RDNA3.5 (gfx1151) 的 LDS（本地数据共享）容量和带宽有限——128 行 × 8 warp 的 tile 导致 LDS staging 停顿，降到 64×4 后 tile 刚好适配 LDS 带宽，减少停顿，吞吐提升 ~38%。

## 次要嫌疑人：FA staging 优化（61f2f2d）

### 改动

旧提交 `61f2f2d`（"avoid full-cache TurboQuant flash-attention staging"）：

```
ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-cpu/quants.h
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn-tile.cuh
ggml/src/ggml-cuda/fattn-vec.cuh
ggml/src/ggml-cuda/fattn.cu
```

### 作用

TurboQuant 的 flash-attention 在长序列时传入了过大的 staging buffer，导致 cache 抖动。改 restricted 了 staging 大小。**对长上下文 prefill 有叠加收益**。

如果移植 MMQ workgroup 调参后 pp512 仍未达到 1000+，再考虑移植此提交。

## 步骤

### 第 1 步：创建分支

```bash
cd /root/reasonix/专六/fusion-rocmfpx-flashnext
git -C src checkout -b prefill-opt
```

### 第 2 步：查看旧 rocmfpx 的参考实现

```bash
# 查看旧提交 e391458 的完整 diff
cd /root/finetuning-workspace/workspace/llama-rocmfpx
git show e391458 -- ggml/src/ggml-cuda/mmq.cuh

# 查看旧版 mmq.cuh 的完整代码
cat ggml/src/ggml-cuda/mmq.cuh | head -250
```

### 第 3 步：在新版 mmq.cuh 中找到对应位置

```bash
cd /root/reasonix/专六/fusion-rocmfpx-flashnext/src
# 找 get_mmq_y / get_mmq_nwarps 或 RDNA3_5 相关函数
grep -n 'RDNA3_5\|get_mmq_y\|get_mmq_nwarp\|mmq_y\|I\b.*=.*64\|nwarps.*=.*4' ggml/src/ggml-cuda/mmq.cuh
```

### 第 4 步：应用调参

在新版 `mmq.cuh` 中，找到 gfx1151 (RDNA3.5) 分支，将 `mmq_y`（或等价 template 参数 `I`）设为 64、`nwarps` 设为 4。

**注意**：新版上游可能已将代码重构为模板参数形式（`I` 等），不能直接复制 e391458 的补丁，需理解逻辑后手动移植。

### 第 5 步：编译

```bash
cd /root/reasonix/专六/fusion-rocmfpx-flashnext
./build.sh
```

### 第 6 步：基准验证

```bash
# 35B Q6_K（主要验证目标：pp512 ≥ 1000）
LD_LIBRARY_PATH=/opt/llama-rocmfpx-next/lib /opt/llama-rocmfpx-next/bin/llama-bench \
  -m /data/models/qwen/Heretic/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q6_K.gguf \
  -p 512 -n 128 -ngl 99 -fa on -lm dio

# Flash-Next Q2_K_XL（回归检查：不能变慢）
LD_LIBRARY_PATH=/opt/llama-rocmfpx-next/lib /opt/llama-rocmfpx-next/bin/llama-bench \
  -m /data/models/qwen38-flash-next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  -p 512 -n 128 -ngl 99 -fa on -lm dio
```

### 第 7 步：如果 pp512 仍未达到 1000+

移植 FA staging 优化（`61f2f2d`）：

```bash
cd /root/finetuning-workspace/workspace/llama-rocmfpx
git show 61f2f2d
```

该提交改 5 个文件，主要涉及 `fattn-*.cuh` 和 `fattn.cu`。相对独立，移植冲突风险低。

## 验收标准

- [x] 35B Q6_K: pp512 ≥ 1000 t/s, tg128 ≥ 50 t/s
- [x] Flash-Next Q2_K_XL: pp512 不降级（当前 ~360），tg128 不降级（当前 ~25）
- [x] 融合构建的 MTP decode 不受影响（~115 t/s warm）
- [x] 编译通过，W4A4=OFF（铁律），无警告
- [x] 改动仅限 kernel 层（ggml-cuda/），不涉及模型/加载层

## 验收结果（2026-05-26, prefill-opt 分支 f765ce1）

同构建 A/B（仅切换 mmq 两个文件, 6b999d4 vs f765ce1, 串行 r4/r5）:

| 模型 | 指标 | 6b999d4（基线） | f765ce1（本优化） | 结论 |
|---|---|---|---|---|
| 35B Q6_K (Heretic MTP) | pp512 | 795.65 ± 11.58 | **1091.10 ± 17.35** | **+37.2%, 达标 (≥1000)** |
| 35B Q6_K | tg128 | 52.50 ± 0.40 | 52.65 ± 0.35 | 持平 (≥50 ✓) |
| Flash-Next Q2_K_XL | pp512 | 277.25 ± 10.00 | 311.78 ± 12.65 | +12.5%, 无退化 |
| Flash-Next Q2_K_XL | tg128 | 23.90 ± 0.56 | 24.83 ± 0.13 | +3.9%, 无退化 |

MTP decode（run-server.sh + bench.sh, Q2_K_XL + mtp draft）: warm 解码 159.3/160.3/159.5 t/s（接受率 1.00）, 高于 ~115 t/s 参考线, 无退化。
交叉参考: 旧 rocmfpx 构建同机 pp512=1056-1068; GLM-4.7-Flash Q4_K_XL 新旧构建 pp512 1298 vs 1301 / tg128 53.2 vs 54.0 持平。

注: OKR 早前记录的 Flash-Next ~360/25 为不同环境态下的测量; 严格同构建 A/B 显示补丁对 Q2_K_XL 同样是净提升。

### 实际根因（与首要嫌疑人不同）

e391458 的 workgroup geometry（64 行/4 warp → 新版 CASE 表 128 线程行 occupancy 2→1, 保留）只贡献小头。**大头是旧仓库私有的 routed-MoE ncols_picker**: `ncols_max` 是每专家最坏宽度, 而 MoE MMQ 应按典型每专家宽度 `ncols_typical = ncols_dst / nchannels_x`（35B: 4096/256=16）选 N-tile。缺此补丁时 MUL_MAT_ID 固定选 J=128, 每个块为 ~16 活跃行读取整窗 Y, 专家权重流带宽塌到 1/3（微基准 4.3ms→恒定 65GB/s）。移植后微基准 1.367ms/6284 GF/s, 与旧构建持平。

## 已知坑

1. **W4A4=OFF 铁律**：`GGML_HIP_ROCMI4_W4A4=ON` 会毁掉 prefill 注意力计算（模型答非所问）。`build.sh` 已固定 OFF。
2. **TTY 竞态**：llama-server 输出不能直写 tmux pane，否则触发 fit-params 假 OOM。测试用 `run-server.sh`（已重定向到文件）。
3. **新上游 mmq.cuh 已重构**：旧 e391458 的补丁不能直接 `git apply`，需要手动移植。
4. **mmq.cuh 的改动只影响 MMQ 路径**：如果 35B Q6_K 走的是非 MMQ 路径（如 MFMA），则需同时调对应的 kernel 参数。运行 `llama-bench` 时日志会显示 kernel 选择。