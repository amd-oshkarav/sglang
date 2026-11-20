# Week 47 P7, the wide launches at 32 rows, against P6, monokernel plugin

Images: `qwen38-mono-ab:05b9dd9fab-1120` (arm a, P6) and `qwen38-mono-ab:a05e4dac1b-1120` (arm b, P7). The images are local only. ITT builds each image on the node from the `dockerfile` key of its arm and does not push it to a registry.

This branch is SGLang `28e5763574` (2026-11-20), the SGLang of the base image, with no SGLang change. The images change only the sgl-amd-mono plugin. The arms are in [`qwen38/spec_decode_mi355/mono_plugin/wide32/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/6cdfc1ad1d914e6db34b97f97f2864be02976d82/qwen38/spec_decode_mi355/mono_plugin/wide32) of AMD-AGI/amd-silo-ai-tiger-teams-sglang.

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-20) |
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
| b (P7) | [`a05e4dac1b`](https://github.com/omirosh/sgl-amd-mono/commit/a05e4dac1b1490af6f9bf06e4aab676769e20099) | `q38v3/9-wide32` | Not open yet |

- `a05e4dac1b` is `05b9dd9fab` and 4 commits:
  - `6553a56`: the wide decode kernels of [vllm#67421](https://github.com/vllm-project/vllm/pull/67421) and [vllm#67678](https://github.com/vllm-project/vllm/pull/67678), with each wait under the plugin guard.
  - `8bb884f`: the 32-row decode and verify steps run one wide launch for each layer, after SGLang's attention or GDN.
  - `1806bcb`: the kernel test and the e2e checks at 32 rows.
  - `a05e4da`: the e2e test passes `--enable-amd-monokernel` when SGLang has it. The kernels are the kernels of `1806bcb`.

## Image

1. [`ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/6cdfc1ad1d914e6db34b97f97f2864be02976d82/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) starts from `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261120@sha256:d548ce1ad6479ec8004521e1964a464ebecd8b1f28b4625cf05b88646264a386` (SGLang `28e5763574`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/6cdfc1ad1d914e6db34b97f97f2864be02976d82/qwen38/spec_decode_mi355/mono_plugin/wide32/gen_yamls.py) makes the arms `wide32/{nomtp,nextn}_{a,b}_{r1,r2,g2,g3}.yaml` and the gates `wide32/{nomtp,nextn}_b_gate.yaml`. The environment is the `environment` block of these YAMLs.
- All arms: `--enable-amd-monokernel` (this SGLang refuses `--monokernel-backend amd`: [sglang#45389](https://github.com/sgl-project/sglang/pull/45389) renamed it), `--enforce-shared-experts-fusion`, page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8. The GDN projections are online MXFP4, with no Hadamard rotation.
- Without MTP: 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows.
- NEXTN 3/1/4 with `--enable-linear-replayssm-spec`: 16 running requests, CUDA graphs at 1, 2, 4, 8 and 16 requests.

## Runs

The MLflow runs are in the section `## MLflow runs` of [`wide32/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/olga/qwen38/spec_decode_mi355/mono_plugin/wide32/PLAN.md).

## Warning

- The plugin reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin.
- P6 and P7 refuse Hadamard-rotated GDN projections. [#46059](https://github.com/sgl-project/sglang/pull/46059) (open on 2026-11-20) turns the rotation on by default. With an SGLang that has it, set `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0`.
