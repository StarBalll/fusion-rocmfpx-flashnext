# fusion-rocmfpx-flashnext

**ROCmFPX(ROCm 优化内核)+ drluoto Flash-Next MTP 草稿头 + prefill-opt MMQ 调参** 的融合构建,
面向 AMD Strix Halo (gfx1151)。

上游与内核说明见 [README-ROCmFPX.md](README-ROCmFPX.md);构建/基准工具与全部实测数据在 [fusion/](fusion/)。

## 最终结果

同机构建 A/B(仅切换 mmq 两个文件,`6b999d4` 基线 vs `prefill-opt` 补丁版,GPU 串行):

| 模型 | 指标 | 基线 | **本构建** | 变化 |
|---|---|---|---|---|
| 35B Q6_K(Heretic MTP) | pp512 | 795.65 t/s | **1091.10 ± 17.35 t/s** | **+37.2%,达标(>=1000)** |
| 35B Q6_K | tg128 | 52.50 t/s | 52.65 ± 0.35 t/s | 持平 |
| Flash-Next Q2_K_XL(176B) | pp512 | 277.25 t/s | 311.78 ± 12.65 t/s | +12.5% |
| Flash-Next Q2_K_XL | tg128 | 23.90 t/s | 24.83 ± 0.13 t/s | +3.9% |
| Flash-Next + MTP 草稿 | 热启解码 | - | **159~160 t/s(接受率 1.00)** | 最大亮点 |
| Flash-Next + MTP 草稿 | 冷启解码 | 33.9 t/s* | 32.5 t/s | 持平 |
| GLM-4.7-Flash Q4_K_XL(交叉验证) | pp512 / tg128 | 旧构建 1301 / 54.0 | 1298 / 53.2 | 持平,无退化 |

\* 冷启数据来自 drluoto 分支对照。参考系:stock b10752 pp512 约 794;旧 rocmfpx (c49ebdb) 约 1056~1068。

**核心优化**:routed-MoE MMQ 的 N-tile 选择应使用典型每专家宽度
(`ncols_dst / nchannels_x`),而不是最坏情形 `ncols_max`。缺失该逻辑时
`MUL_MAT_ID` 固定选 J=128,每个块为 ~16 个活跃行读整窗 Y,专家权重流带宽
掉到约 1/3。完整分析:[fusion/PREFILL_1000_OKR.md](fusion/PREFILL_1000_OKR.md)。

## 本机环境(全部数据实测自该机器)

| 项 | 值 |
|---|---|
| 平台 | AMD Ryzen AI MAX+ 395 (Strix Halo),Radeon 8060S iGPU |
| GPU | gfx1151(RDNA3.5,Wave32,统一显存 121 GiB 可见 126976 MiB) |
| 系统 | Ubuntu 24.04.4 LTS,内核 7.0.0-14-generic |
| HIP / ROCm | HIP 7.2.53211;hipcub/rocprim 头取自 /opt/rocm-extras/opt/rocm-7.2.4 |
| 工具链 | cmake 3.28.3,gcc 15.2.0 |
| 构建参数 | `GGML_HIP=ON` `GGML_VULKAN=OFF` `GGML_HIP_ROCMI4_W4A4=OFF`(铁律)`CMAKE_HIP_ARCHITECTURES=gfx1151` Release |
| 安装前缀 | /opt/llama-rocmfpx-next(动态链接,运行需 `LD_LIBRARY_PATH=/opt/llama-rocmfpx-next/lib`) |
| 测试模型 | 35B Q6_K: /data/models/qwen/Heretic/ 下 Qwen3.6-35B-A3B MTP-Preserved Q6_K;Flash-Next Q2_K_XL(3 分卷)+ mtp-Q8_0 草稿头 |

## 快速开始

```bash
git clone git@github.com:StarBalll/fusion-rocmfpx-flashnext.git
cd fusion-rocmfpx-flashnext

./fusion/build.sh                    # 构建 + 安装(可加 --skip-patches 跳过上游校验)
./fusion/run-server.sh               # 启动 Flash-Next + MTP 测试服务(日志入文件)
./fusion/verify.sh 8132              # 健康/模型/答题质量自检
./fusion/bench.sh                    # 服务端解码 + prefill 对比
./fusion/bench/gate.sh <35B-Q6_K.gguf>   # prefill 回归门槛(默认 >= 1000 t/s)
```

## 基准复现(llama-bench 部分)

```bash
export LD_LIBRARY_PATH=/opt/llama-rocmfpx-next/lib
B=/opt/llama-rocmfpx-next/bin/llama-bench

# 35B prefill / decode
$B -m <35B-Q6_K.gguf> -p 512 -n 0 -ngl 99 -fa on -r 4
$B -m <35B-Q6_K.gguf> -p 0 -n 128 -ngl 99 -fa on -r 4

# Flash-Next Q2_K_XL(分卷模型给第一个文件)
$B -m <Q2_K_XL-00001-of-00003.gguf> -p 512 -n 0 -ngl 99 -fa on -r 4
```

服务端 MTP 解码用 `fusion/run-server.sh` + `fusion/bench.sh`(冷/热两轮,
输出解码 t/s 与草稿接受率)。

## 测试数据存档

[fusion/bench/data/](fusion/bench/data/) 按类别存放原始输出:

- `final-ab/` — 同构建 A/B 的 llama-bench 记录(35B 与 Flash-Next 全套)
- `op-trace/` — 真实 pp512 计算图逐 op 耗时(`test-backend-ops --test-file`)
- `graph-export/` — 新旧构建导出的 GGML 图 + op 枚举翻译表(0.11 <-> 0.22)
- `microbench/` — MMID / FA(head 256)/ SSM 独立微基准输出

注意:`--test-file` 的后端过滤要用设备名 `ROCm0`(不是 CUDA0);导出文件只在
同 op 枚举的构建间可移植,跨版本用 `graph-export/enum_map.json` 翻译。

## 目录速览

```
fusion/
  build.sh              一键构建
  run-server.sh         MTP 测试服务(输出重定向,防 TTY 竞态)
  verify.sh / bench.sh  服务自检 / 服务端基准
  prefill-bench.py      长 prompt prefill 基准
  bench/gate.sh         pp512 回归门槛
  bench/data/           全部原始测试数据
  env/                  Dockerfile + 环境说明
  PREFILL_1000_OKR.md   1000 t/s 目标完整验收记录
```

## 已知坑

1. `GGML_HIP_ROCMI4_W4A4` 必须 OFF(有损实验路径,会毁掉 prefill 注意力)。
2. llama-server 输出绝不能直写 TTY(触发 fit-params 假 OOM),一律重定向文件。
3. GPU 基准严格串行,不叠加两个 20GB+ 模型。
4. `rocm.sh` 会重置 `LD_LIBRARY_PATH`,追加安装库路径必须在其后。
