# collector_261128_qwen_3.8_mono_nofse

The 20261127 nightly with the Qwen3.8 monokernel stack of sgl-amd-mono plus the no-FSE support at `23fe99e1a1`, with no SGLang change. No-FSE means a server without `--enforce-shared-experts-fusion`: SGLang keeps the bf16 shared expert.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_nofse@sha256:1f2968c8f3d32bad58f7f35901ab8dbb4c673a62e53998f1e320181fa770f9e4` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/nofse/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/weekend_1128/nofse), arm `n`. The MLflow parent runs are in `weekend_1128/PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) | base image | none |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `23fe99e1a1c61c9ebc3a93ed4c5dc3772c326b12` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/18-nofse), branch `q38v3/18-nofse` | the stack plus 3 commits, section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## sgl-amd-mono

| Commits | PR | PR head used | State (2026-11-28) |
| --- | --- | --- | --- |
| up to `fb59229` | the stack: [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) (P6), [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159), [sgl-amd-mono#174](https://github.com/sgl-project/sgl-amd-mono/pull/174), [sgl-amd-mono#180](https://github.com/sgl-project/sgl-amd-mono/pull/180), the rotation guard and D2 | the heads of [`collector_261128_qwen_3.8_mono_stack`](https://github.com/amd-oshkarav/sglang/tree/collector_261128_qwen_3.8_mono_stack) | open and draft |
| `adabdda` | none yet | | the 16-row split launches (front16) run without FSE: `layer_block` of the bf16 shared expert runs their MLP half |
| `2b8c598` | none yet | | the 32-row wide launches (P7) run without FSE: the router matrix also holds the shared gate and up rows and the shared gate, and the least busy CTAs run the shared `down_proj` |
| `23fe99e` | none yet | | without FSE, `layer_block` runs the steps of 3, 6 and 8 rows (`LAYER_WIDTHS_F0`), K2 the others of 1 to 8 rows |

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9`, and the run's `--max-running-requests` and CUDA graph sizes.
- The arms of this image run without `--enforce-shared-experts-fusion`. With the flag, the image runs the code of the stack.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_nofse https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_nofse collector/
```

## Warnings

- K1v is not in this image. The combined image [`collector_261128_qwen_3.8_mono_all`](https://github.com/amd-oshkarav/sglang/tree/collector_261128_qwen_3.8_mono_all) has K1v, item 1, Hadamard-32 and the no-FSE support.
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image.
