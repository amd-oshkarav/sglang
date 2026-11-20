# Week 47 stack_p2 and sgl-amd-mono#154, monokernel plugin

Image: `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_18_11_qwen_3.8_stack_p2_mono154@sha256:f90b49864d99b8067312ca592d7bc68940712658b9759ee36aac57ba84a82260`. Its local tag on the node is `qwen38-collector:1811-stack-p2-mono154`.

This branch is [`collector_09_11_qwen_3.8_stack_p2`](https://github.com/amd-oshkarav/sglang/tree/collector_09_11_qwen_3.8_stack_p2) (`3590bb919d`) plus two code commits. The other commits of the branch change only this README. The code commits give the stack the sgl-amd-mono plugin of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154). That plugin runs the tail of each 16-row decode or verify step as one `layer_block` launch.

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

- The rows to #39595 are the rows of `collector_09_11_qwen_3.8_stack_p2`. The commits come from the layer files in [`qwen38/spec_decode_mi355/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4fffe20a57e0e6ab2ed01be1e6bec3bd4ecce608/qwen38/spec_decode_mi355) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`.
- The plugin of #154 calls `register(..., model_type=...)`. The main base `e4d4a9a7de` of this branch is one day older than #46036, so its `register()` has no `model_type`. Without `5cdfd1933b`, the plugin fails at load.
- With `5cdfd1933b`, the sgl-amd-mono contract test (`test/unit/test_contract.py`) passes for `sgl_amd_mono/__init__.py` and `qwen38_a95b_mxfp4/` on this tree. The contract fails for other models. The plugin imports their specs only for their architectures, so they do not load on a Qwen3.8 server.

## AITER

The stack uses AITER main [`9059673612`](https://github.com/ROCm/aiter/commit/90596736129ccc0bcf720686cf202067a11f110d) (2026-10-26), with one edit:

| Edit | Stack item | AITER file | Change | Source files | On AITER main |
| --- | --- | --- | --- | --- | --- |
| a4w4 GEMM rows, N 4608, K 8192 | G-1 | `aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv`, a new file | Adds 26 tuned a4w4 blockscale GEMM shapes: gfx950, 256 CUs, M 1 to 16384, N 4608, K 8192. That is the GDN `in_proj_qkvz` at TP8. `apply.py` stops the build when AITER already has the file or one of its shapes. | Dockerfile [`g1_a4w4_pr7441/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/Dockerfile), edit script [`apply.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/apply.py), data [`qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv), all at `441dc8a6` | Yes. [aiter#7441](https://github.com/ROCm/aiter/pull/7441) merged the file on 2026-11-05 (`ec91f61f0c`). On an AITER with that commit, `apply.py` stops the build. Drop this layer there. |

- The image has no P-1b AITER edit ([`p1b_stage2_fp8/patch_aiter.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/p1b_stage2_fp8/patch_aiter.py)). The edit has no effect on Qwen3.8 (README of `collector_09_11_qwen_3.8_stack_p2`). P-1b needs only `AITER_FLYDSL_STAGE2_FP8=1`.
- FlyDSL is 0.4.0, the pin of AITER `9059673612`. #154 was tested with AITER `285579b621`, which also pins FlyDSL 0.4.0.

**Warning:** do not move AITER past [`0e072ca50e`](https://github.com/ROCm/aiter/commit/0e072ca50ee78c8e7c686a41a16c03524f92c276) for a copy of this stack. [aiter#5988](https://github.com/ROCm/aiter/pull/5988) puts the Q256 ASM prefill kernel into the E 513 rows at 8192, 16384 and 32768 tokens. The kernel does not read `AITER_FLYDSL_STAGE2_FP8`. On this stack, no MTP, on one node, the rows give 98.4% throughput, 93.4% TTFT and 97.5% TPOT ([`p1b_aiter5988/AITER5988_WALKTHROUGH.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/p1b_aiter5988/AITER5988_WALKTHROUGH.md)).

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

**Warning:** do not move the plugin to the later heads of #154. The head `05b9dd9fab` (2026-11-19) uses `linear_attn_backend.state_layer_cache` ([sgl-amd-mono#147](https://github.com/sgl-project/sgl-amd-mono/pull/147)). SGLang main has it in `hybrid_linear_attn_backend.py`, but this branch does not. Keep `410ca7872d`.

## Image

- Image: `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_18_11_qwen_3.8_stack_p2_mono154@sha256:f90b49864d99b8067312ca592d7bc68940712658b9759ee36aac57ba84a82260`. The registry has it. Its local tag on the node is `qwen38-collector:1811-stack-p2-mono154`.
- Base image: `lmsysorg/sglang:v0.5.22-rocm10-mi35x@sha256:1438dd8b9000adaefb5828edb1cc5713992815d4296f4ad1a1e98dee4e9c7e5f`.
- Start commit of the branch: `3590bb919d`, the code of `collector_09_11_qwen_3.8_stack_p2`.

The build script [`collector_stack_p2/build.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/collector_stack_p2/build.sh) builds the three layers on the node. Then it pushes the images of steps 2 and 3. The file links of the steps and of the source build are at commit [`441dc8a6`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/441dc8a613db126057cd3471a9f27d3e916c78f3) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`. The node of job 297596 used this commit for the build.

1. [`collector26_mono/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/collector26_mono/Dockerfile) builds SGLang, AITER and sgl-amd-mono from source on the base image.
   - `build.sh` gives `SGLANG_BRANCH=collector_09_11_qwen_3.8_stack_p2` and `SGLANG_REV=3590bb919ddb321b61b2075b5ed2b37fc67a4b74`.
   - The defaults of the Dockerfile give AITER `90596736129ccc0bcf720686cf202067a11f110d` and sgl-amd-mono `c8b3850aef498814f189bf63edc0f6543c138c94` (`omirosh/sgl-amd-mono`).
   - The layer applies no patch or diff. The SGLang changes are the commits of the branch.
   - The result is `qwen38-collector:0911-src` (`sha256:a86e44e97e258b83838cfbe0638abaed82b0dc790cb88542ded92eb5800ce87f`). This image is local only. The registry does not have it.
2. [`g1_a4w4_pr7441/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/Dockerfile) adds the G-1 AITER edit on `qwen38-collector:0911-src` (section "AITER").
   - The Dockerfile copies [`qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv) and the edit script [`apply.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/apply.py). Then it runs `apply.py`.
   - The result is `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_09_11_qwen_3.8_stack_p2@sha256:cc1594a9ad1388343635bd9fe189a2ef6e22f02863bcb6c685b2adcfd1cc422e`, the image of `collector_09_11_qwen_3.8_stack_p2`. Its local tag is `qwen38-collector:0911-stack-p2`.
3. [`collector_stack_p2/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/collector_stack_p2/Dockerfile) adds the code of this branch on `qwen38-collector:0911-stack-p2`.
   - It checks out `afea911d7bb8ee0f1a03d0b999d4a126daef2f5e`, the default `SGLANG_REV`, in `/sgl-workspace/sglang`.
   - It installs the plugin pin of `python/pyproject_other.toml`: `410ca7872d`, the head of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) on 2026-11-18.
   - It runs the contract test and the Qwen3.8 spec test of sgl-amd-mono. The build log gives 26 passed.
   - The result is the image of this branch (`sha256:f90b4986…`).

No layer comes from an ITT `dockerfile` key. Thus no file of the image is only in ITT or MLflow.

Job 297596 (`crsuse2-m2m-v3-073`, 2026-11-19 00:18 UTC) built and pushed the digests above. The arms of that job run these images. The first build of job 297397 pushed other digests to the same tags, and the second push replaced them. Job 298735 (`crsuse2-m2m-v3-252`) pulled the digests above with [`build.sh` at `601a2cea`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/601a2cea7bd8323d8b4e1b9e3e0d592c5609009f/qwen38/spec_decode_mi355/collector_stack_p2/build.sh) and `PULL=1`. It did not build again. The build logs are in `~/oob/logs/qwen38-collector-p2/collector_build/` on the Crusoe login.

Against `3590bb919d`, the branch changes two Python files, the plugin pin in `python/pyproject_other.toml`, and the README. It changes no compiled code. Thus step 3 gives the same code as a build of this branch from source. To build this branch from source:

1. Build [`collector26_mono/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/collector26_mono/Dockerfile) on the base image with these build arguments:
   - `SGLANG_BRANCH=collector_18_11_qwen_3.8_stack_p2_mono154`
   - `SGLANG_REV=afea911d7bb8ee0f1a03d0b999d4a126daef2f5e`
   - `SGL_AMD_MONO_COMMIT=410ca7872d8029a5532c39fe1930d9a1b23ac77a`
   - `SGL_AMD_MONO_TESTS="qwen38_a95b or common or __init__"`
2. Build [`g1_a4w4_pr7441/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/441dc8a613db126057cd3471a9f27d3e916c78f3/qwen38/spec_decode_mi355/g1_a4w4_pr7441/Dockerfile) on the image of step 1. Set `BASE_IMAGE` to that image.

No script or YAML of this repo builds this path. Thus it is not tested.

## Runs

The runs are in [`qwen38/spec_decode_mi355/collector_stack_p2/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/collector_stack_p2/PLAN.md): the same-node comparison against `qwen38-stack-p2:9059673612` and `collector_09_11_qwen_3.8_stack_p2`, no MTP and MTP.

The full sweeps of this image ran in job 298735 on `crsuse2-m2m-v3-252`, 2026-11-19. The evals ran in job 297596 on `crsuse2-m2m-v3-073`, 2026-11-19:

| Run | MLflow | Result |
| --- | --- | --- |
| No MTP, 12 cells | [718975bd](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/718975bdf2e44f91b676dd526d45fa3b) | 97.5% of the B300 20261026 output throughput. The week 46 best run d07515fc gives 96.0% |
| MTP, 14 cells | [5121a31e](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/5121a31eb03a4b8da9de278f7681d924) | 110.1% of the B300 20261026 MTP output throughput. The week 46 best run 5f9e0490 gives 108.6% |
| No MTP, evals | [d3416d63](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/d3416d6337f74d17b0c70e7b54352cd4) | GSM8K, GPQA Diamond, NIAH: inside the noise of the week 46 scores |
| MTP, evals | [aa13f8de](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/aa13f8de6c884610b833ce76654df706) | GSM8K, GPQA Diamond, NIAH: inside the noise of the week 46 scores |

With MTP, the server can stall for 9 to 13 s before a large prefill batch. In 5121a31e, the stalls fall into 8k c8 and 8k c128. Compare the TPOT of those cells, not the throughput.

The server needs `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. The YAMLs have the full command: [`stack_261105/nonmtp/stack_p2.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/stack_261105/nonmtp/stack_p2.yaml) (no MTP) and [`stack_261105/stack_ar_p2.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/stack_261105/stack_ar_p2.yaml) (MTP).
