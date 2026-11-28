# collector_261128_qwen_3.8_mono_p6

The 20261127 nightly with the Qwen3.8 monokernels of P6 ([sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154)) at `d28e8399dc`, with no SGLang change. It is the reference arm of the stack.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_p6@sha256:c2a51ff9044b8ec39c9ca9a6598475346140581cb9d02d72953a000059fa79f8` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4468dde6/qwen38/spec_decode_mi355/mono_plugin/weekend_1128), arm `p`. The MLflow parent runs are in its `PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) | base image | none |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `d28e8399dc42efd179d91e9dce1f58ef2eb29e40` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono) | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## sgl-amd-mono

| Commit | PR | PR head used | State (2026-11-28) |
| --- | --- | --- | --- |
| `d28e8399dc` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) | `d28e8399dc42efd179d91e9dce1f58ef2eb29e40` | open |

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9 --max-running-requests 64`.
- `--enforce-shared-experts-fusion` is optional. P6 runs K1 and K2 with and without the fused shared expert.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_p6 https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_p6 collector/
```

## Warnings

- P6 moves while it is open. A new head gets a new branch.
