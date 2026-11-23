# Week 47 P7, the wide launches at 32 rows, against front16, monokernel plugin

Images of the bench (ITT): `qwen38-mono-ab:e7cee030a3-1121` (arm a, front16), `qwen38-mono-ab:343bdf93db-1121` (arm b, P7) and `qwen38-mono-ab:05b9dd9fab-1121` (arm p, P6, only the 60k / 600 c64 cell). Images of GPQA and SWE-50 (`p1_acc`): `qwen38-mono-test:e7cee030a3-1121`, `qwen38-mono-test:343bdf93db-1121` and `qwen38-mono-test:05b9dd9fab-1121`. The images are local only. ITT and `tests.sh` build each image on the node and do not push it to a registry.

This branch is SGLang `c93e78483b` (2026-11-21), the SGLang of the base image, with no SGLang change. The images change only the sgl-amd-mono plugin. The arms are in [`qwen38/spec_decode_mi355/mono_plugin/wide32/f16/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/2ce0370f3fc9ba45b720a1d271c051882919c499/qwen38/spec_decode_mi355/mono_plugin/wide32/f16) of AMD-AGI/amd-silo-ai-tiger-teams-sglang.

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-22) |
| --- | --- | --- | --- |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quark MXFP4 fused shared expert | Yes, through the base (`0cd2a63b36`) | Yes, merged |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | MXFP4 GDN projections, online | Yes, through the base (`3145c7ae66`) | Yes, merged |
| [#46980](https://github.com/sgl-project/sglang/pull/46980) | The state route of the linear attention backend (`state_layer_cache`) | Yes, through the base (`306a699a60`) | Yes, merged |
| [#50759](https://github.com/sgl-project/sglang/pull/50759) | The gfx950 AITER pin `581579ab5` | Yes, through the base (`b5a4fca3a6`) | Yes, merged |

## AITER

The images use the AITER of the base image, [`581579ab51`](https://github.com/ROCm/aiter/commit/581579ab51ca26ddd4a08a6aa6aa983a78aaf276), with no edit. FlyDSL is the FlyDSL of the base image.

## sgl-amd-mono

The plugin comes from [`omirosh/sgl-amd-mono`](https://github.com/omirosh/sgl-amd-mono). The images do not use the pin of `python/pyproject_other.toml`.

| Arm | Plugin commit | Branch | Pull request |
| --- | --- | --- | --- |
| p (P6) | [`05b9dd9fab`](https://github.com/omirosh/sgl-amd-mono/commit/05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68) | `q38v3/6-layer-block` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) |
| a (front16) | [`e7cee030a3`](https://github.com/omirosh/sgl-amd-mono/commit/e7cee030a346cbad8350df5cb85a849ee9ec7fe3) | `q38v3/10-front16` | [sgl-amd-mono#174](https://github.com/sgl-project/sgl-amd-mono/pull/174) |
| b (P7) | [`343bdf93db`](https://github.com/omirosh/sgl-amd-mono/commit/343bdf93db773066ff62e5005dd348e9307999f5) | `q38v3/9-wide32` | [sgl-amd-mono#180](https://github.com/sgl-project/sgl-amd-mono/pull/180) (draft) |

The head of [sgl-amd-mono#174](https://github.com/sgl-project/sgl-amd-mono/pull/174) is `8046962`. It is `e7cee030a3` and 2 commits that change only an e2e test (`0dcc499`), the README and a docstring (`8046962`). So arm a has the package code of the head.

`343bdf93db` is `8046962` and 7 commits:

- `fa6aff2`: the tails of a 32-row step (`attn` and `gdn_back`) run as the wide launches of [vllm#67421](https://github.com/vllm-project/vllm/pull/67421) and [vllm#67678](https://github.com/vllm-project/vllm/pull/67678), for decode and NEXTN verify.
- `a2cd757`: the kernel test checks the wide launches on 8 GPUs at 32 rows (`check_wide32.py`).
- `89bc304`: the wide fault check gives the peers a longer deadline.
- `d5a5c0d`: check mode limits the route flips of each launch kind.
- `b16026d`: in a decode step of 32 rows, each GDN layer runs one `gdn` launch, with the GDN front.
- `9c9d000`: the kernel test checks the GDN front of the wide launches.
- `343bdf9`: the combine after a TP exchange norms the residual sum before its bf16 rounding, as SGLang's `fused_add_rms_norm`, K1, K2 and `layer_block` do.

The head of [sgl-amd-mono#180](https://github.com/sgl-project/sgl-amd-mono/pull/180) is `e6cfeb1`. It is `343bdf93db` and 2 commits that change only check mode, its kernel test and the README: `67ae31e` (`check_wide32.py`) and `e6cfeb1` (`check.py`, the check-mode part of `spec.py`). So arm b has the code that the bench and the NEXTN verify steps run at the head.

## Image

1. [`ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/2ce0370f3fc9ba45b720a1d271c051882919c499/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) starts from `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261121@sha256:3530e2fc84734759474758be484dcb8185f9bcba44b8d54b312d0aa0f21a0319` (SGLang `c93e78483b`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

ITT builds the bench images from the `dockerfile` key of each arm. [`tests.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/2ce0370f3fc9ba45b720a1d271c051882919c499/qwen38/spec_decode_mi355/mono_plugin/tests.sh) builds the `p1_acc` images from the same Dockerfile (`TESTS_TAG=1121`).

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/2ce0370f3fc9ba45b720a1d271c051882919c499/qwen38/spec_decode_mi355/mono_plugin/wide32/gen_yamls.py) (mode `f16`) makes the arms `wide32/f16/nomtp_{a,b}_{r1,r2,g2,g3}.yaml` and `wide32/f16/nomtp_p_{l1,l2}.yaml`. The environment is the `environment` block of these YAMLs.
- All arms: `--enable-amd-monokernel`, `--enforce-shared-experts-fusion`, page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8, no MTP. The GDN projections are online MXFP4, with no Hadamard rotation.
- 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows. The split launch runs in the decode steps of 16 rows, and the wide launches in the decode steps of 32 rows (P7 only).
- NEXTN (Change 4): [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5dedda2756dd8bd0a8ddd7723e7699d9c2dbc3df/qwen38/spec_decode_mi355/mono_plugin/wide32/gen_yamls.py) (mode `f16nx`) makes the arms [`wide32/f16nx/nextn_{a,b}_{r1,r2,g2,g3}.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/5dedda2756dd8bd0a8ddd7723e7699d9c2dbc3df/qwen38/spec_decode_mi355/mono_plugin/wide32/f16nx), with the same images and server flags, plus `--speculative-algorithm NEXTN` with 3 steps, top-k 1 and 4 draft tokens, and `--enable-linear-replayssm-spec`. 16 running requests, CUDA graphs at 1, 2, 4, 8 and 16 requests. The wide launches run in the verify steps of 32 rows (8 requests).
- GPQA and SWE-50: [`p1_acc/serve.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/2ce0370f3fc9ba45b720a1d271c051882919c499/qwen38/spec_decode_mi355/mono_plugin/p1_acc/serve.sh) (cases `p6n`, `front16` and `p7`) with the same server flags. SWE-50 keeps the radix cache.

## Runs

The MLflow runs and the GPQA and SWE-50 results are in the section `## Change 3` of [`wide32/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/olga/qwen38/spec_decode_mi355/mono_plugin/wide32/PLAN.md). The NEXTN runs (job 305840) are in the section `## Change 4`.

## Warning

- The plugin reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin.
- P6, front16 and P7 refuse Hadamard-rotated GDN projections. [#46059](https://github.com/sgl-project/sglang/pull/46059) (open on 2026-11-22) turns the rotation on by default. With an SGLang that has it, set `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0`.
- The runs of Change 2 (`collector_20_11_qwen_3.8_wide32`, P7 at `7cf2e23` on P6) used the combine that norms the rounded sum. Their numbers do not apply to `343bdf93db`.
