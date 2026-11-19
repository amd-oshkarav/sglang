# Week 47 stack_p2 and sgl-amd-mono#154, monokernel plugin

Image: `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_18_11_qwen_3.8_stack_p2_mono154@sha256:f90b49864d99b8067312ca592d7bc68940712658b9759ee36aac57ba84a82260`. Its local tag on the node is `qwen38-collector:1811-stack-p2-mono154`.

This branch is [`collector_09_11_qwen_3.8_stack_p2`](https://github.com/amd-oshkarav/sglang/tree/collector_09_11_qwen_3.8_stack_p2) (`3590bb919d`) plus two commits. They give the stack the sgl-amd-mono plugin of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154). That plugin runs the tail of each 16-row decode or verify step as one `layer_block` launch.

The stack runs with AITER main `9059673612` and one AITER edit (G-1). AITER is not in this repo. The section "AITER" gives the edit.

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
| [#46036](https://github.com/sgl-project/sglang/pull/46036) | Register monokernel specs by model type | mono154 | `5cdfd1933b` | Yes, merged 2026-10-27 |
| - | Pin sgl-amd-mono to the head of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) | mono154 | `9ded272c50` | - |

- The rows to #39595 are the rows of `collector_09_11_qwen_3.8_stack_p2`. The commits come from the layer files in [`qwen38/spec_decode_mi355/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`.
- The plugin of #154 calls `register(..., model_type=...)`. The main base `e4d4a9a7de` of this branch is one day older than #46036, so its `register()` has no `model_type`. Without `5cdfd1933b`, the plugin fails at load.
- With `5cdfd1933b`, the sgl-amd-mono contract test (`test/unit/test_contract.py`) passes for `sgl_amd_mono/__init__.py` and `qwen38_a95b_mxfp4/` on this tree. The contract fails for other models. The plugin imports their specs only for their architectures, so they do not load on a Qwen3.8 server.

## AITER

The stack uses AITER main [`9059673612`](https://github.com/ROCm/aiter/commit/90596736129ccc0bcf720686cf202067a11f110d) (2026-10-26), with one edit:

| Edit | Stack item | Change | Source |
| --- | --- | --- | --- |
| a4w4 GEMM rows, N 4608, K 8192 | G-1 | Adds `aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv`. [aiter#7441](https://github.com/ROCm/aiter/pull/7441) merged this file on 2026-11-05. | `g1_a4w4_pr7441/` |

- The image has no P-1b AITER edit (`p1b_stage2_fp8/patch_aiter.py`). The edit has no effect on Qwen3.8 (README of `collector_09_11_qwen_3.8_stack_p2`). P-1b needs only `AITER_FLYDSL_STAGE2_FP8=1`.
- FlyDSL is 0.4.0, the pin of AITER `9059673612`. #154 was tested with AITER `285579b621`, which also pins FlyDSL 0.4.0.

**Warning:** do not move AITER past [`0e072ca50e`](https://github.com/ROCm/aiter/commit/0e072ca50ee78c8e7c686a41a16c03524f92c276) for a copy of this stack. [aiter#5988](https://github.com/ROCm/aiter/pull/5988) puts the Q256 ASM prefill kernel into the E 513 rows at 8192, 16384 and 32768 tokens. The kernel does not read `AITER_FLYDSL_STAGE2_FP8`. On this stack, no MTP, on one node, the rows give 98.4% throughput, 93.4% TTFT and 97.5% TPOT (`p1b_aiter5988/AITER5988_WALKTHROUGH.md`).

## sgl-amd-mono

The plugin is [`omirosh/sgl-amd-mono` @ `410ca7872d`](https://github.com/omirosh/sgl-amd-mono/commit/410ca7872d8029a5532c39fe1930d9a1b23ac77a), the head of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) on 2026-11-18. `python/pyproject_other.toml` pins it.

| Part | Commits | State (2026-11-18) |
| --- | --- | --- |
| sgl-amd-mono main | to `56669ca` ([sgl-amd-mono#90](https://github.com/sgl-project/sgl-amd-mono/pull/90), the fused shared expert) | Merged |
| [sgl-amd-mono#92](https://github.com/sgl-project/sgl-amd-mono/pull/92): MXFP4 GDN projections, Hadamard refusal | `5516f20` to `a4c94bb` | Open |
| [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154): `layer_block` at 16 rows | `a32391c`, and the test commits to `410ca78` | Open, draft |

- The earlier collectors pin the fork commit `c8b3850aef`. This branch moves the plugin to the upstream code of K1 and K2, plus #92 and #154.
- #154 is a draft and can get a rebase. The tag `collector_18_11_qwen_3.8_stack_p2_mono154` of `omirosh/sgl-amd-mono` keeps `410ca7872d`.
- `layer_block` runs only with `--enforce-shared-experts-fusion`. A step of 16 rows then runs it: no MTP at 9 to 16 requests, NEXTN 3/1/4 at 4 requests. The server log gives "Qwen3.8 monokernel: ... and 2 layer_block launches".
- Start the server with `--enable-amd-monokernel`.

## Image

The image is the image of `collector_09_11_qwen_3.8_stack_p2` plus one layer. The job of `qwen38/spec_decode_mi355/collector_stack_p2/` builds the images on the node and pushes them (`collector_stack_p2/build.sh`).

1. `collector26_mono/Dockerfile` builds `collector_09_11_qwen_3.8_stack_p2` (`3590bb919d`) from source on `lmsysorg/sglang:v0.5.22-rocm10-mi35x@sha256:1438dd8b9000adaefb5828edb1cc5713992815d4296f4ad1a1e98dee4e9c7e5f`, with AITER `9059673612` and sgl-amd-mono `c8b3850aef`.
2. `g1_a4w4_pr7441/Dockerfile` adds the G-1 CSV. The result is `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_09_11_qwen_3.8_stack_p2` (`sha256:cc1594a9ad1388343635bd9fe189a2ef6e22f02863bcb6c685b2adcfd1cc422e`).
3. `collector_stack_p2/Dockerfile` checks out this branch in `/sgl-workspace/sglang`, installs the plugin pin of `python/pyproject_other.toml` and runs its unit tests. The result is the image of this branch (`sha256:f90b49864d99b8067312ca592d7bc68940712658b9759ee36aac57ba84a82260`).

Job 297596 (`crsuse2-m2m-v3-073`, 2026-11-19 00:18 UTC) built and pushed the digests above. The arms of that job run these images. The first build of job 297397 pushed other digests to the same tags, and the second push replaced them.

The branch changes only Python files against `3590bb919d`. Thus step 3 gives the same code as a build of this branch from source. For a build from source, give `collector26_mono/Dockerfile` the branch, its head, `SGL_AMD_MONO_COMMIT=410ca7872d8029a5532c39fe1930d9a1b23ac77a` and `SGL_AMD_MONO_TESTS="qwen38_a95b or common or __init__"`, then add step 2.

## Runs

The runs are in `qwen38/spec_decode_mi355/collector_stack_p2/PLAN.md`: the same-node comparison against `qwen38-stack-p2:9059673612` and `collector_09_11_qwen_3.8_stack_p2`, no MTP and MTP.

The server needs `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. The YAMLs have the full command: `stack_261105/nonmtp/stack_p2.yaml` (no MTP) and `stack_261105/stack_ar_p2.yaml` (MTP).
