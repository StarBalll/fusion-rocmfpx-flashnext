# fusion-rocmfpx-flashnext (ROCmFPX + Flash-Next MTP)

Fusion build of:
- **ROCmFPX main** - ROCm-optimized MMQ/FA kernel family for gfx1151 (RDNA3.5 / Strix Halo)
- **drluoto strix-halo-flash-next** - 5 commits adding the qwen4exp NextN/MTP draft head
  (Flash-Next) on top of upstream llama.cpp
- **prefill-opt** - routed-MoE MMQ ncols_picker + RDNA3.5 tile geometry port from the
  older rocmfpx tree

This repository root IS the llama.cpp source tree (ROCmFPX main + cherry-picks + the
`prefill-opt` MMQ commits). The `fusion/` directory holds the build/bench tooling and
recorded benchmark data.

## Results (gfx1151, 35B Q6_K, same-build A/B, serial runs)

| model | metric | baseline | **this build** | delta |
|---|---|---|---|---|
| 35B Q6_K (Heretic MTP) | pp512 | 795.65 | **1091.10** | **+37.2%** |
| 35B Q6_K | tg128 | 52.50 | 52.65 | flat |
| Flash-Next Q2_K_XL | pp512 | 277.25 | 311.78 | +12.5% |
| Flash-Next Q2_K_XL | tg128 | 23.90 | 24.83 | +3.9% |
| Flash-Next + MTP | warm decode | - | **159-160 t/s (acc 1.00)** | best |

The main win is the routed-MoE ncols_picker: MMQ N-tile selection must use the typical
per-expert width (`ncols_dst / nchannels_x`), not the worst-case `ncols_max`. Without it
`MUL_MAT_ID` picks J=128 and expert weight streaming drops to ~1/3 bandwidth.
Details and full data: [PREFILL_1000_OKR.md](PREFILL_1000_OKR.md),
[bench/data/](bench/data/).

## Build

Requirements: ROCm HIP toolchain (hipcc), gfx1151 target, optional hipcub/rocprim headers.

```bash
git clone https://github.com/StarBalll/fusion-rocmfpx-flashnext.git
cd fusion-rocmfpx-flashnext

# full build: verify cherry-picks vs upstream -> cmake -> make -> install
./fusion/build.sh

# skip upstream sync (offline / already synced)
./fusion/build.sh --skip-patches
```

W4A4 is hard-wired OFF (`-DGGML_HIP_ROCMI4_W4A4=OFF`); it is an experimental lossy path
that breaks prefill attention. Do not enable it.

## Run / verify

```bash
# start the test server (Flash-Next Q2_K_XL + MTP draft head), log to /tmp/llama-fusion.out
./fusion/run-server.sh

# self-check: health / alias / answer quality / MTP accept rate
./fusion/verify.sh 8132

# server-side decode + prefill benchmark
./fusion/bench.sh

# long-prompt prefill benchmark (unique content per round, no KV hits)
python3 fusion/prefill-bench.py 8132 2
```

Edit `MODEL`/`MTP` in `fusion/run-server.sh` to point at your GGUFs.

## Benchmark data

`bench/data/` contains the recorded evidence:

- `final-ab/` - same-build A/B llama-bench outputs (baseline vs prefill-opt, r4/r5)
- `op-trace/` - per-op timings of the real pp512 graph via `test-backend-ops --test-file`
- `graph-export/` - exported GGML graphs (new/old builds) + op-enum translation map
- `microbench/` - standalone MMID / FA / SSM micro-benchmark outputs

`test-backend-ops --test-file` note: the backend filter must match the device name
(`ROCm0` on this stack), and test files are only portable between builds with the same
ggml op enum - `graph-export/enum_map.json` translates between the old (0.11) and new
(0.22) enums.

## Known pitfalls

1. `GGML_HIP_ROCMI4_W4A4=OFF` is a hard rule (see above).
2. Never let llama-server write directly to a TTY; timing changes trigger a spurious
   OOM in fit-params. `run-server.sh` redirects to a file.
3. GPU benchmarks are serial - never overlap 20GB+ models.
4. Dynamic linking needs `LD_LIBRARY_PATH=/opt/llama-rocmfpx-next/lib`; `rocm.sh`
   resets that variable, so append after sourcing it.
