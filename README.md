# Week 47 P6 with the monokernels off and on, monokernel plugin

Image: `qwen38-mono-ab:05b9dd9fab-1117`. The image is local only. ITT builds it on the node from the `dockerfile` key of each arm and does not push it to a registry.

This branch is SGLang `c67db7c762` (2026-11-17), the SGLang of the base image, with no SGLang change. The image changes only the sgl-amd-mono plugin. The image code and the arms are in [`qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) and [`qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117) of [AMD-AGI/amd-silo-ai-tiger-teams-sglang](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang) (branch `olga`). These two links use the commit [`d514b041`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/d514b041c99aecd691cd883beb303a76f8015ea0). The Crusoe login clone had this commit when the `r1` and `r2` arms started (job 299113).

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

**Warning:** without build arguments, `ab/Dockerfile` builds from `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261016` with the plugin `83009d8911`. Set `BASE_IMAGE` and `PLUGIN_SHA` to the values of the `dockerfile` key of an arm YAML.

1. [`ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) starts from the build argument `BASE_IMAGE`. The arm YAMLs set it to `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261117@sha256:09b8237296024285e6286ffb54d75c3af6d59342bbbb3558dea56bed86798814` (SGLang `c67db7c762`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`. The arm YAMLs set `PLUGIN_SHA` to `05b9dd9fab`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

ITT logs the Dockerfile in MLflow. The copy of the first arm is [`ab/Dockerfile`](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/b71dfaace9a94259abc5ff5c96bc3775/artifacts?selectedArtifact=b71dfaace9a94259abc5ff5c96bc3775%3Aab%2FDockerfile&o=4032599967441609) in the child run `b71dfaac` of `off_nomtp_r1`. It is the same as the file at `d514b041`.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- The arms are `p6_offon_1117/{off,on}_{nomtp,nextn}_r{1,2}.yaml`. The environment is the `environment` block of these YAMLs.
- The GSM8K rounds g2 and g3 use the same image, server flags and environment: `p6_offon_1117/{off,on}_{nomtp,nextn}_g{2,3}.yaml`. These arms run the concurrency 1 cell and GSM8K only.
- All arms: `--enforce-shared-experts-fusion`, page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8.
- The on arms add `--monokernel-backend amd`.
- Without MTP: 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows.
- NEXTN 3/1/4 with `--enable-linear-replayssm-spec`: 16 running requests, CUDA graphs at 1, 2, 4, 8 and 16.

Each arm YAML has the `dockerfile` key and the server command. The links use the commit of the Crusoe login clone at the start of each arm.

| Job | Commit | Arm YAMLs |
| --- | --- | --- |
| 299113 | [`d514b041`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/d514b041c99aecd691cd883beb303a76f8015ea0) | [`off_nomtp_r1`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nomtp_r1.yaml), [`on_nomtp_r1`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nomtp_r1.yaml), [`on_nomtp_r2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nomtp_r2.yaml), [`off_nomtp_r2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nomtp_r2.yaml), [`off_nextn_r1`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nextn_r1.yaml), [`on_nextn_r1`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nextn_r1.yaml), [`on_nextn_r2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nextn_r2.yaml), [`off_nextn_r2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/d514b041c99aecd691cd883beb303a76f8015ea0/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nextn_r2.yaml) |
| 300492 | [`ec04a120`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/ec04a120db80aece7fff429fe221f804803844c9) | [`on_nomtp_g2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/ec04a120db80aece7fff429fe221f804803844c9/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nomtp_g2.yaml) |
| 300492 | [`210ba217`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/210ba217bf662032a99eb2c9c0d75316d50a53c2) | [`off_nomtp_g2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nomtp_g2.yaml), [`off_nomtp_g3`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nomtp_g3.yaml), [`on_nomtp_g3`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nomtp_g3.yaml), [`on_nextn_g2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nextn_g2.yaml), [`off_nextn_g2`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nextn_g2.yaml), [`off_nextn_g3`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/off_nextn_g3.yaml), [`on_nextn_g3`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/210ba217bf662032a99eb2c9c0d75316d50a53c2/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/on_nextn_g3.yaml) |

## Runs

The MLflow runs are in the section `## MLflow runs` of [`p6_offon_1117/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/mono_plugin/p6_offon_1117/PLAN.md).

## Warning

The plugin `05b9dd9fab` reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin, for example `5bb7ecc2c2` of the 20261028 image.
