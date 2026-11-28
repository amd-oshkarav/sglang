# collector_261128_qwen_3.8_mono_og

The image of `collector_261128_qwen_3.8_mono_all` plus two sgl-amd-mono commits, with no SGLang change. The plugin is at `3cf33b156d` (`q38v3/20-og`):

- og: the launch after each full-attention layer (K2, `layer_block`, the wide `attn` launch) multiplies the attention output gate in. SGLang's `fused_sigmoid_mul` kernel before `o_proj` does not run.
- `layer_block` without the fused shared expert also runs at 2 rows (`LAYER_WIDTHS_F0 = (2, 3, 6, 8)`).

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_og@sha256:15cec1f2a405b0e29e8f85e98a3c3cf3ee85073c38c97bd4fba8e9c40ccc9334` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/allfse/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/weekend_1128), arm `g`. The MLflow parent runs are in `weekend_1128/PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) | base image | none |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `3cf33b156d60dc6c86d90e160e84453bd69d7b62` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/20-og), branch `q38v3/20-og` | `q38v3/19-all` plus 2 commits, section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## sgl-amd-mono

The branch is `a76ad03` (`q38v3/19-all`, the plugin of `collector_261128_qwen_3.8_mono_all` plus one test commit) with these commits:

| Commit | PR | What it does |
| --- | --- | --- |
| `c7e169e` | none yet (og) | `_core` swaps `qwen3_5.fused_sigmoid_mul` while SGLang's attention runs, so the attention returns the ungated `o_proj` input and the gate. The K2 and `layer_block` keys get the field `og`. The attention layers get `og=True`, the GDN layers `og=False`. The wide `attn` key gets `og=True`. `GatedCore` multiplies in fp32 and rounds once, as SGLang's `fused_sigmoid_mul`. The swap refuses `quant=True` and `round_sigmoid=True`. |
| `3cf33b1` | none yet (D2 without FSE) | `LAYER_WIDTHS_F0 = (2, 3, 6, 8)`: with item 1 of vllm-project/vllm#61271, `layer_block` without FSE is 5.3% faster than K2 at 2 rows |

The checks of `3cf33b1` on the 20261127 nightly (`weekend_1128/PLAN.md`, section 6.8):

- `test_og`: 20 K2 and `layer_block` cases against SGLang's `fused_sigmoid_mul`, 20 bit-equal on rank 0.
- `test_wide32`: all 12 chain cases pass with the gate in the `attn` launch.
- The e2e test: 6 of 6 pass (check mode with FSE off and on, check mode on the ReplaySSM ring, and `TestMonokernelDecode`).

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9`, and the run's `--max-running-requests` and CUDA graph sizes.
- NEXTN: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec`.
- The image runs with and without `--enforce-shared-experts-fusion`.
- `SGLANG_AMD_MONO_K1V=0` turns K1v off.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_og https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_og collector/
```

## Warnings

- og needs SGLang's `fused_sigmoid_mul` in `sglang.srt.models.qwen3_5`, with `quant=False` and `round_sigmoid=False`. Another SGLang call stops the server with a RuntimeError.
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image.
