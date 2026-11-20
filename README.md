# Week 47 layer_block without the fused shared expert, FSE off and on, monokernel plugin

Images: `qwen38-mono-ab:760dd51e38-1119` (arm a) and `qwen38-mono-ab:b8c981e11e-1119` (arms off and b). The images are local only. ITT builds each image on the node from the `dockerfile` key of its arm and does not push it to a registry.

This branch is SGLang `b1970bc473` (2026-11-19), the SGLang of the base image, with no SGLang change. The images change only the sgl-amd-mono plugin. The arms are in [`qwen38/spec_decode_mi355/mono_plugin/lb_nofse_offon_1119/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/d11dbe93afe0ecdf9d2f21b6973e260337e4e7fb/qwen38/spec_decode_mi355/mono_plugin/lb_nofse_offon_1119) of AMD-AGI/amd-silo-ai-tiger-teams-sglang.

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

| Arms | Plugin commit | Branch | Pull request |
| --- | --- | --- | --- |
| a | [`760dd51e38`](https://github.com/omirosh/sgl-amd-mono/commit/760dd51e383f366e893d174221fc3f508b775f44) | `q38v3/7-hadamard` | [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), draft |
| off and b | [`b8c981e11e`](https://github.com/omirosh/sgl-amd-mono/commit/b8c981e11edb3133a7965509a7029d525e454d91) | `q38v3/8-layer-block-nofse` | [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159), draft |

- [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159) is on top of [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156). It runs `layer_block` at 16 rows also without the fused shared expert.
- The off arms run the image of b without `--enable-amd-monokernel`. Then SGLang serves each step with its own forward.

## Image

1. [`ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d11dbe93afe0ecdf9d2f21b6973e260337e4e7fb/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) starts from `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261119@sha256:55917fb2b7fc4bf8068a02afd5bd70573c83c50465879a60cc2277fafba9769a` (SGLang `b1970bc473`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d11dbe93afe0ecdf9d2f21b6973e260337e4e7fb/qwen38/spec_decode_mi355/mono_plugin/lb_nofse_offon_1119/gen_yamls.py) makes the arms `lb_nofse_offon_1119/fse{off,on}_{nomtp,nextn}_{off,a,b}_r{1,2}.yaml`. FSE on has no off arm. The environment is the `environment` block of these YAMLs.
- All arms: page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8. The GDN projections are online MXFP4, with no Hadamard rotation.
- The FSE on arms add `--enforce-shared-experts-fusion`. The a and b arms add `--enable-amd-monokernel`. This SGLang refuses `--monokernel-backend amd`: [sglang#45389](https://github.com/sgl-project/sglang/pull/45389) renamed it.
- Without MTP: 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows.
- NEXTN 3/1/4 with `--enable-linear-replayssm-spec`: 16 running requests, CUDA graphs at 1, 2, 4, 8 and 16 requests.

## Runs

The MLflow runs are in the section `## MLflow runs` of [`lb_nofse_offon_1119/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/olga/qwen38/spec_decode_mi355/mono_plugin/lb_nofse_offon_1119/PLAN.md).

## Warning

- The plugin reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin.
- [#46059](https://github.com/sgl-project/sglang/pull/46059) (open on 2026-11-20) turns the Hadamard rotation of the GDN projections on by default. With an SGLang that has it, set `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0` for these arms.
