# Week 46 stack_p2, monokernel plugin

Image: `qwen38-stack-p2:9059673612`. The image is a local tag on the node. It is not in a registry.

This branch has the SGLang code of that image. The branch starts at [`collector_26_10_qwen_3.8_add_rms_norm_mono_plugin`](https://github.com/amd-oshkarav/sglang/tree/collector_26_10_qwen_3.8_add_rms_norm_mono_plugin) (`396b0bcc69`). That branch is [`collector_26_10_qwen_3.8_mono_plugin`](https://github.com/amd-oshkarav/sglang/tree/collector_26_10_qwen_3.8_mono_plugin) (`493cd5e719`) plus the add-RMSNorm commits. Each later commit is one item of the stack.

The stack runs with AITER main `9059673612` and two AITER edits. AITER is not in this repo. The section "AITER" gives the edits.

## SGLang

| Pull request | Change | Stack item | On this branch | On SGLang main (2026-11-18) |
| --- | --- | --- | --- | --- |
| [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP | Q1 | Yes, through `493cd5e719` | No, open |
| [#40708](https://github.com/sgl-project/sglang/pull/40708) | 512-expert softmax MoE through AITER topk_gating | Q1 | Yes, through `493cd5e719` | No, open |
| [#40771](https://github.com/sgl-project/sglang/pull/40771) | QuickReduce before custom all-reduce | Q1 | Yes, through `493cd5e719` | No, open |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quark MXFP4 fused shared expert | Q1 | Yes, through the main base `e4d4a9a7de` | Yes |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | MXFP4 GDN projections, online | Q1 | Yes, through `493cd5e719` | No, open |
| [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | FlyDSL INT4 QuickReduce | Q1 | Yes, through `493cd5e719` | No |
| [#47219](https://github.com/sgl-project/sglang/pull/47219) | add-RMSNorm in the FlyDSL INT4 QuickReduce | P-1a | `356cccd165`, `a20dfda8af` | No, open |
| [#47433](https://github.com/sgl-project/sglang/pull/47433) | ReplaySSM with one pipeline stage on gfx950 | M-3 | `7a18b9e7b3` | No, open |
| [#45529](https://github.com/sgl-project/sglang/pull/45529) | Fused self-attn prepare (norm, RoPE, gate) | D-2a | `e367b1a5d6` | No, open |
| [#41683](https://github.com/sgl-project/sglang/pull/41683) | Strided GDN prefill views on HIP | P-4a | `bb6ce61297` | Yes, merged 2026-11-17 |
| [#47269](https://github.com/sgl-project/sglang/pull/47269) | GDN verify mask only with PP stable rows | M-5a | `a3d2001e34` | No, open |
| [#45555](https://github.com/sgl-project/sglang/pull/45555) | Small-batch MTP verify attention over all CUs | M-5 | `30eda13df5` | Yes, merged 2026-10-27 |
| [#39595](https://github.com/sgl-project/sglang/pull/39595) | FlyDSL GDN prefill backend, port to `493cd5e719` | P-2 | `e41260f0f8` | No, open |

The commits come from the layer files in [`qwen38/spec_decode_mi355/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`: `p1a_qr_add_rmsnorm/files`, `m3_rssm_pr47433/patch_sglang.py`, `d2a_pr45529/pr45529.diff`, `p4a_pr41683/patch_sglang.py`, `m5_verify_glue/patch_sglang.py` and `pr45555.diff`, `p2_pr39595/pr39595_493.diff`. The P-1a files are the same as the files of `396b0bcc69`.

## AITER

The stack uses AITER main [`9059673612`](https://github.com/ROCm/aiter/commit/90596736129ccc0bcf720686cf202067a11f110d) (2026-10-26), with two edits:

| Edit | Stack item | Change | Source |
| --- | --- | --- | --- |
| Stage 2 FP8 tiers | P-1b | `aiter/ops/flydsl/moe_kernels.py`: `if force_reduce or fp8_reduce:` becomes `if force_reduce:`. With `AITER_FLYDSL_STAGE2_FP8=1`, only the prefill reduce tiers write FP8 rows. | `p1b_stage2_fp8/patch_aiter.py` |
| a4w4 GEMM rows, N 4608, K 8192 | G-1 | Adds `aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv`. [aiter#7441](https://github.com/ROCm/aiter/pull/7441) merged this file on 2026-11-05. | `g1_a4w4_pr7441/` |

**Warning:** do not move AITER past [`0e072ca50e`](https://github.com/ROCm/aiter/commit/0e072ca50ee78c8e7c686a41a16c03524f92c276) for a copy of this stack. [aiter#5988](https://github.com/ROCm/aiter/pull/5988) puts the Q256 ASM prefill kernel into the E 513 rows at 8192, 16384 and 32768 tokens. The kernel does not read `AITER_FLYDSL_STAGE2_FP8`. On this stack, no MTP, on one node, the rows give 98.4% throughput, 93.4% TTFT and 97.5% TPOT (`p1b_aiter5988/AITER5988_WALKTHROUGH.md`).

## sgl-amd-mono

The plugin is [`omirosh/sgl-amd-mono` @ `c8b3850aef`](https://github.com/omirosh/sgl-amd-mono/commit/c8b3850aef498814f189bf63edc0f6543c138c94), as on `collector_26_10_qwen_3.8_mono_plugin`. Start the server with `--enable-amd-monokernel`.

## Image

The image is a stack of layers on the Q1 image `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_26_qr_flydsl_int4_mono_plugin` (`sha256:8b65d34a`). `493cd5e719` of this repo is the SGLang of Q1.

1. `p1b_stage2_fp8/Dockerfile` makes `qwen38-p1b:9059673612` (P-1b).
2. `stack_261105/build.sh` adds the layers P-1a, M-3, G-1, D-2a, P-4a and M-5. The result is `qwen38-stack:9059673612`.
3. `p2_pr39595/Dockerfile` adds P-2. The result is `qwen38-stack-p2:9059673612`. ITT builds this layer from the `dockerfile` key of the YAML.

The image of the 2026-11-09 build on `crsuse2-m2m-v3-105` is `sha256:b588607e`.

## Runs

- No MTP: [d07515fc](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/d07515fc226d4484819ecd6d017312f4), YAML `stack_261105/nonmtp/stack_p2.yaml`.
- MTP: `stack_261105/STACK_AR_P2.md`, YAML `stack_261105/stack_ar_p2.yaml`.

The server needs `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. The YAMLs have the full command.
