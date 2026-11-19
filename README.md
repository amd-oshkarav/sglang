# Week 47 P6 with the monokernels off and on, monokernel plugin

Image: `qwen38-mono-ab:05b9dd9fab-1117`. The image is local only. ITT builds it on the node from the `dockerfile` key of each arm and does not push it to a registry.

This branch is SGLang `c67db7c762` (2026-11-17), the SGLang of the base image, with no SGLang change. The image changes only the sgl-amd-mono plugin. The image code and the arms are in `qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile` and `qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/` of [AMD-AGI/amd-silo-ai-tiger-teams-sglang](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117) (branch `olga`).

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-19) |
| --- | --- | --- | --- |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quark MXFP4 fused shared expert | Yes, through the base (`0cd2a63b36`) | Yes, merged |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | MXFP4 GDN projections, online | Yes, through the base (`3145c7ae66`) | Yes, merged |
| [#46980](https://github.com/sgl-project/sglang/pull/46980) | The state route of the linear attention backend (`state_layer_cache`) | Yes, through the base (`306a699a60`) | Yes, merged |
| [#50759](https://github.com/sgl-project/sglang/pull/50759) | The gfx950 AITER pin `581579ab5` | Yes, through the base (`b5a4fca3a6`) | Yes, merged |

## AITER

The image uses the AITER of the base image, [`581579ab51`](https://github.com/ROCm/aiter/commit/581579ab51ca26ddd4a08a6aa6aa983a78aaf276), with no edit. FlyDSL is the FlyDSL of the base image.

## sgl-amd-mono

The plugin comes from [`omirosh/sgl-amd-mono`](https://github.com/omirosh/sgl-amd-mono). The image does not use the pin of `python/pyproject_other.toml` (`a04ceab159`).

| Arms | Plugin commit | Branch | Pull request |
| --- | --- | --- | --- |
| Off and on | [`05b9dd9fab`](https://github.com/omirosh/sgl-amd-mono/commit/05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68) | `q38v3/6-layer-block` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154), draft |

- `05b9dd9fab` is sgl-amd-mono `main` `36c385e` and the 5 commits of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154).
- The off arms run the same image without `--monokernel-backend`. Then SGLang serves each step with its own forward.

## Image

1. `ab/Dockerfile` starts from `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261117@sha256:09b8237296024285e6286ffb54d75c3af6d59342bbbb3558dea56bed86798814` (SGLang `c67db7c762`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- The arms are `p6_offon_1117/{off,on}_{nomtp,nextn}_r{1,2}.yaml`. The environment is the `environment` block of these YAMLs.
- All arms: `--enforce-shared-experts-fusion`, page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8.
- The on arms add `--monokernel-backend amd`.
- Without MTP: 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows.
- NEXTN 3/1/4 with `--enable-linear-replayssm-spec`: 16 running requests, CUDA graphs at 1, 2, 4, 8 and 16.

## Runs

The MLflow runs are in the section `## MLflow runs` of `p6_offon_1117/PLAN.md`.

## Warning

The plugin `05b9dd9fab` reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin, for example `5bb7ecc2c2` of the 20261028 image.
