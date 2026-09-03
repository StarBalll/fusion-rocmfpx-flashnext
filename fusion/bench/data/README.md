# Recorded benchmark evidence (gfx1151 / Strix Halo, serial GPU runs)

- final-ab/          same-build A/B llama-bench (baseline 6b999d4 vs prefill-opt f765ce1)
  - 35b_base.txt / final_35b.txt / opt_verify.txt   35B Q6_K pp512+tg128
  - nw_bench.txt                                    no-warmup cross-check (r4)
  - fn_q2xl*.txt / flashnext*.txt                   Flash-Next Q2_K_XL + GLM-4.7-Flash
- op-trace/          per-op us of the real pp512 graph (test-backend-ops --test-file)
- graph-export/      exported GGML graphs (pp512) + op-enum translation map (old 0.11 <-> new 0.22)
- microbench/        standalone MMID / FA(head 256) / SSM micro-benchmark outputs

Context: PREFILL_1000_OKR.md in fusion/. Numbers quoted in fusion/README.md come from these files.
