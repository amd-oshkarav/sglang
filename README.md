# Week 48 D2, layer_block at 2, 3 and 5 to 8 rows, against P6, monokernel plugin

Images: `qwen38-mono-ab:05b9dd9fab-1121` (arm a, P6) and `qwen38-mono-ab:7b1bdb21d6-1121` (arm b, the D2 width choice). The images are local only. ITT builds each image on the node from the `dockerfile` key of its arm and does not push it to a registry.

This branch is SGLang `c93e78483b` (2026-11-21), the SGLang of the base image, with no SGLang change. It starts from the collector branch `collector_21_11_qwen_3.8_front16` (`aeff511a04`), which has the same base image. The images change only the sgl-amd-mono plugin. The arms are in [`qwen38/spec_decode_mi355/mono_plugin/d2/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/18bc04f95c61a5a85be0887144b589d7448fbcbd/qwen38/spec_decode_mi355/mono_plugin/d2) of AMD-AGI/amd-silo-ai-tiger-teams-sglang.

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-21) |
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
| a (P6) | [`05b9dd9fab`](https://github.com/omirosh/sgl-amd-mono/commit/05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68) | `q38v3/6-layer-block` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) |
| b (D2) | [`7b1bdb21d6`](https://github.com/omirosh/sgl-amd-mono/commit/7b1bdb21d627505a092bd9d176d0b58b69006a3a) | `q38v3/11-d2` | Not open yet |

`7b1bdb21d6` is `05b9dd9fab` and 3 commits:

- `6a4f8aa`: the kernel test times K2 against `layer_block` at 1 to 8 rows (`check_k2.py --time`). Test only.
- `702d370`: a `timeline` build flag of K2 and `layer_block` (a clock mark at the end of each phase), and the threshold options of `check_k2.py`. The plain builds do not change.
- `7b1bdb2`: with the fused shared expert, the spec runs `layer_block` in place of K2 at 2, 3 and 5 to 8 rows (`LAYER_WIDTHS`). K2 stays at 1 and 4 rows. The e2e check mode also checks `layer_block` at 3 and 8 rows.

## Image

1. [`ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/18bc04f95c61a5a85be0887144b589d7448fbcbd/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) starts from `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261121@sha256:3530e2fc84734759474758be484dcb8185f9bcba44b8d54b312d0aa0f21a0319` (SGLang `c93e78483b`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/18bc04f95c61a5a85be0887144b589d7448fbcbd/qwen38/spec_decode_mi355/mono_plugin/d2/gen_yamls.py) makes the arms `d2/{nomtp,nextn}_{a,b}_{r1,r2}.yaml` from the P6 arms `p6_offon_1117/on_{nomtp,nextn}_r1.yaml`. The environment is the `environment` block of these YAMLs.
- All arms: `--enable-amd-monokernel`, `--enforce-shared-experts-fusion`, page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8. The GDN projections are online MXFP4, with no Hadamard rotation.
- No MTP: 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows. NEXTN 3/1/4 with `--enable-linear-replayssm-spec`: 16 running requests, CUDA graphs at 1, 2, 4, 8 and 16 rows.
- The arms differ only in the steps of 2 and 8 rows: no MTP at c2 and c8, NEXTN at c2.

## Runs

The MLflow runs are in the section `## MLflow runs` of [`d2/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/olga/qwen38/spec_decode_mi355/mono_plugin/d2/PLAN.md).

## Warning

- The plugin reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin.
- P6 and D2 refuse Hadamard-rotated GDN projections. [#46059](https://github.com/sgl-project/sglang/pull/46059) turns the rotation on by default. With an SGLang that has it, set `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0`.
- Without `--enforce-shared-experts-fusion`, arm b runs K2 at 1 to 8 rows, as arm a does.
