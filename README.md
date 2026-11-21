# Week 47 front16, the split GDN launch at 16 rows, against P6, monokernel plugin

Images: `qwen38-mono-ab:05b9dd9fab-1121` (arm a, P6) and `qwen38-mono-ab:e7cee030a3-1121` (arm b, front16). The images are local only. ITT builds each image on the node from the `dockerfile` key of its arm and does not push it to a registry.

This branch is SGLang `c93e78483b` (2026-11-21), the SGLang of the base image, with no SGLang change. The images change only the sgl-amd-mono plugin. The arms are in [`qwen38/spec_decode_mi355/mono_plugin/front16/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/52e744e2ddd8a3c69c438bbb6a0ba246d2e7f018/qwen38/spec_decode_mi355/mono_plugin/front16) of AMD-AGI/amd-silo-ai-tiger-teams-sglang.

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
| b (front16) | [`e7cee030a3`](https://github.com/omirosh/sgl-amd-mono/commit/e7cee030a346cbad8350df5cb85a849ee9ec7fe3) | `q38v3/10-front16` | Not open yet |

`e7cee030a3` is `05b9dd9fab` and 7 commits:

- `d894ff1`: the wide decode kernels of [vllm#67421](https://github.com/vllm-project/vllm/pull/67421) and [vllm#67678](https://github.com/vllm-project/vllm/pull/67678), with each wait under the plugin guard.
- `46c6c69`: the `gdn` kind of the wide kernels reads and writes SGLang's conv layout and can keep a bf16 SSM state.
- `920dfe8`: the e2e test passes `--enable-amd-monokernel` when SGLang has it.
- `32fe416`: in a decode step of 16 rows, each GDN layer runs one split launch (input norm, `in_proj_qkvz` and `in_proj_ba`, the conv, the recurrence and the gated norm), then `layer_block`.
- `707d297`: the kernel test compares the split launch with K1 at 16 rows.
- `002a789`: the e2e check-mode test needs a split launch check at 16 rows and 2 split launches in the build log.
- `e7cee03`: the input norm of the split launch reads the fp32 sum of the layer input and the residual, as SGLang's `fused_add_rms_norm` and K1 do. The vendored code normed the bf16 sum, as vLLM does.

## Image

1. [`ab/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/52e744e2ddd8a3c69c438bbb6a0ba246d2e7f018/qwen38/spec_decode_mi355/mono_plugin/ab/Dockerfile) starts from `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261121@sha256:3530e2fc84734759474758be484dcb8185f9bcba44b8d54b312d0aa0f21a0319` (SGLang `c93e78483b`, AITER `581579ab51`).
2. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
3. It checks that the plugin registers `Qwen3_5MoeForCausalLM`, and it writes `/etc/sglang-build/manifest.txt`.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 and 3.

## Server flags and environment

- [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/52e744e2ddd8a3c69c438bbb6a0ba246d2e7f018/qwen38/spec_decode_mi355/mono_plugin/front16/gen_yamls.py) makes the arms `front16/nomtp_{a,b}_{r1,r2,g2,g3}.yaml` and the gate `front16/nomtp_b_gate.yaml`. The environment is the `environment` block of these YAMLs.
- All arms: `--enable-amd-monokernel`, `--enforce-shared-experts-fusion`, page 16, FP8 KV, bf16 SSM state, `--disable-radix-cache`, TP8, no MTP. The GDN projections are online MXFP4, with no Hadamard rotation.
- 64 running requests, CUDA graphs at 1, 2, 4, 8, 16, 32 and 64 rows. The split launch runs only in the decode steps of 16 rows.

## Runs

The MLflow runs are in the section `## MLflow runs` of [`front16/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/olga/qwen38/spec_decode_mi355/mono_plugin/front16/PLAN.md).

## Warning

- The plugin reads `state_layer_cache` of [#46980](https://github.com/sgl-project/sglang/pull/46980). An SGLang without it cannot serve this plugin.
- P6 and front16 refuse Hadamard-rotated GDN projections. [#46059](https://github.com/sgl-project/sglang/pull/46059) (open on 2026-11-21) turns the rotation on by default. With an SGLang that has it, set `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0`.
