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

The commits come from the layer files in [`qwen38/spec_decode_mi355/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`: [`p1a_qr_add_rmsnorm/files`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p1a_qr_add_rmsnorm/files), [`m3_rssm_pr47433/patch_sglang.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/m3_rssm_pr47433/patch_sglang.py), [`d2a_pr45529/pr45529.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/d2a_pr45529/pr45529.diff), [`p4a_pr41683/patch_sglang.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p4a_pr41683/patch_sglang.py), [`m5_verify_glue/patch_sglang.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/m5_verify_glue/patch_sglang.py) and [`pr45555.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/m5_verify_glue/pr45555.diff), [`p2_pr39595/pr39595_493.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p2_pr39595/pr39595_493.diff). The P-1a files are the same as the files of `396b0bcc69`.

## AITER

The stack uses AITER main [`9059673612`](https://github.com/ROCm/aiter/commit/90596736129ccc0bcf720686cf202067a11f110d) (2026-10-26), with two edits:

| Edit | Stack item | Change | Source |
| --- | --- | --- | --- |
| Stage 2 FP8 tiers | P-1b | `aiter/ops/flydsl/moe_kernels.py`: `if force_reduce or fp8_reduce:` becomes `if force_reduce:`. With `AITER_FLYDSL_STAGE2_FP8=1`, only the prefill reduce tiers write FP8 rows. | [`p1b_stage2_fp8/patch_aiter.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/40209800abf2e4d8a50997b41c3da2fec2b3f86e/qwen38/spec_decode_mi355/p1b_stage2_fp8/patch_aiter.py) |
| a4w4 GEMM rows, N 4608, K 8192 | G-1 | Adds `aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv`. [aiter#7441](https://github.com/ROCm/aiter/pull/7441) merged this file on 2026-11-05. | [`g1_a4w4_pr7441/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/g1_a4w4_pr7441) |

The P-1b edit is in the image, but it has no effect on Qwen3.8. A copy of this stack does not need it. Only `AITER_FLYDSL_STAGE2_FP8=1` is necessary.

- The Qwen3.8 atomic tiers use `flydsl_moe2_layout_*` kernels. Their launcher `_flydsl_v2_stage2_wrapper` does not read the override. This is the same in AITER `9059673612` and in AITER main.
- The clean same-node A/B without the edit gives the same gain: throughput +6.3%, TTFT -11.1%, TPOT -5.7%. With the edit: +6.2%, -8.7%, -5.3% ([`p1b_stage2_fp8/P1B_STAGE2_FP8_JIRA.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/p1b_stage2_fp8/P1B_STAGE2_FP8_JIRA.md), job 269431).

**Warning:** do not move AITER past [`0e072ca50e`](https://github.com/ROCm/aiter/commit/0e072ca50ee78c8e7c686a41a16c03524f92c276) for a copy of this stack. [aiter#5988](https://github.com/ROCm/aiter/pull/5988) puts the Q256 ASM prefill kernel into the E 513 rows at 8192, 16384 and 32768 tokens. The kernel does not read `AITER_FLYDSL_STAGE2_FP8`. On this stack, no MTP, on one node, the rows give 98.4% throughput, 93.4% TTFT and 97.5% TPOT ([`p1b_aiter5988/AITER5988_WALKTHROUGH.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/p1b_aiter5988/AITER5988_WALKTHROUGH.md)).

## sgl-amd-mono

The plugin is [`omirosh/sgl-amd-mono` @ `c8b3850aef`](https://github.com/omirosh/sgl-amd-mono/commit/c8b3850aef498814f189bf63edc0f6543c138c94), as on `collector_26_10_qwen_3.8_mono_plugin`. Start the server with `--enable-amd-monokernel`.

## Image

The image is a stack of layers on the Q1 image `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_26_qr_flydsl_int4_mono_plugin` (`sha256:8b65d34a`). `493cd5e719` of this repo is the SGLang of Q1.

1. [`p1b_stage2_fp8/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/40209800abf2e4d8a50997b41c3da2fec2b3f86e/qwen38/spec_decode_mi355/p1b_stage2_fp8/Dockerfile) makes `qwen38-p1b:9059673612` ([P-1b](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/40209800abf2e4d8a50997b41c3da2fec2b3f86e/qwen38/spec_decode_mi355/p1b_stage2_fp8)).
2. [`stack_261105/build.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/stack_261105/build.sh) adds the layers [P-1a](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p1a_qr_add_rmsnorm), [M-3](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/m3_rssm_pr47433), [G-1](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/g1_a4w4_pr7441), [D-2a](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/d2a_pr45529), [P-4a](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p4a_pr41683) and [M-5](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/m5_verify_glue). The result is `qwen38-stack:9059673612`.
3. [`p2_pr39595/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p2_pr39595/Dockerfile) adds [P-2](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/p2_pr39595). The result is `qwen38-stack-p2:9059673612`. ITT builds this layer from the `dockerfile` key of the YAML.

The image of the 2026-11-09 build on `crsuse2-m2m-v3-105` is `sha256:b588607e`.

Job 275934 made this build. Each link of the three steps uses the repo commit that the node read for that layer:

- Step 1: [`40209800`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/40209800abf2e4d8a50997b41c3da2fec2b3f86e). ITT built the layer at 09:16 UTC, from the `dockerfile` key of [`stack_261105/base.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/40209800abf2e4d8a50997b41c3da2fec2b3f86e/qwen38/spec_decode_mi355/stack_261105/base.yaml).
- Step 2: [`4fffe20a`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608). `build.sh` ran from 10:57 to 10:59 UTC.
- Step 3: [`4fffe20a`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608). ITT built the layer at 11:00 UTC for [`stack_261105/stack_ar_p2_evals.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355/stack_261105/stack_ar_p2_evals.yaml).

A stack item link goes to the layer folder, with the Dockerfile and the files that it copies.

A build of this branch from source is in the registry: `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_09_11_qwen_3.8_stack_p2@sha256:cc1594a9ad1388343635bd9fe189a2ef6e22f02863bcb6c685b2adcfd1cc422e`. Job 297596 (`crsuse2-m2m-v3-073`, 2026-11-19 00:18 UTC) built it with [`qwen38/spec_decode_mi355/collector_stack_p2/build.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/collector_stack_p2/build.sh):

1. [`collector26_mono/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/collector26_mono/Dockerfile) builds this branch (`3590bb919d`) on `lmsysorg/sglang:v0.5.22-rocm10-mi35x@sha256:1438dd8b9000adaefb5828edb1cc5713992815d4296f4ad1a1e98dee4e9c7e5f`, with AITER `9059673612` and sgl-amd-mono `c8b3850aef`.
2. [`g1_a4w4_pr7441/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/Dockerfile) adds the [G-1 CSV](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv) with [`apply.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/apply.py).

The node read the repo at commit [`441dc8a6`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/441dc8a613db126057cd3471a9f27d3e916c78f3) for this build. The links of this build use that commit.

Against Q1, the pip diff of that image is `apache-tvm-ffi` 0.1.15.post1 to 0.1.16, `pytz` and `tzdata` 2026.5 to 2026.6, and the `sglang` version. The same-node comparison against `qwen38-stack-p2:9059673612` is in [`qwen38/spec_decode_mi355/collector_stack_p2/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/collector_stack_p2/PLAN.md).

## Runs

- No MTP: [d07515fc](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/d07515fc226d4484819ecd6d017312f4), YAML [`stack_261105/nonmtp/stack_p2.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/stack_261105/nonmtp/stack_p2.yaml).
- MTP: [`stack_261105/STACK_AR_P2.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/stack_261105/STACK_AR_P2.md), YAML [`stack_261105/stack_ar_p2.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/stack_261105/stack_ar_p2.yaml).

The server needs `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. The YAMLs have the full command.
