# collector_261128_qwen_3.8_mono_stack

The 20261127 nightly with the full Qwen3.8 monokernel stack of sgl-amd-mono at `fb592293c3`, with no SGLang change.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_stack@sha256:c35a1225ad95b537e49033e9013462dfe205158f51a4e75008c8aec044bb9b81` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/4468dde6/qwen38/spec_decode_mi355/mono_plugin/weekend_1128), arm `s`. The MLflow parent runs are in its `PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) | base image | none |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `fb592293c379af26c09f8db176a31951572524db` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/13-stack), branch `q38v3/13-stack` | 31 commits on P6, section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## sgl-amd-mono

The stack is P6 plus the drafts in this order. Each draft is at its head after the rebase onto P6 (2026-11-28).

| Commits | PR | PR head used | State (2026-11-28) |
| --- | --- | --- | --- |
| up to `d28e8399dc` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) (P6) | `d28e8399dc42efd179d91e9dce1f58ef2eb29e40` | open |
| `378395c` to `879e23b` | [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156) (Hadamard-32) | `879e23b` | draft |
| `5a6743f` to `9e8a2e5` | [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159) (`layer_block` without FSE) | `9e8a2e5` | draft |
| `6ab0097` to `2a1e383` | [sgl-amd-mono#174](https://github.com/sgl-project/sgl-amd-mono/pull/174) (front16, the split launch) | `0388638` | draft |
| `ec21dda` to `f015f96` | [sgl-amd-mono#180](https://github.com/sgl-project/sgl-amd-mono/pull/180) (P7, the wide launches) | `c4f5295` | draft |
| `4b02b2e` | none (stack fix) | | the split and wide launches are not built on a rotated GDN projection |
| `b8a65e0` to `0fec0be` | none yet (D2, branch `q38v3/11-d2`) | `199ce28` | `layer_block` in place of K2 at 2, 3 and 5 to 8 rows, FSE only |
| `fb59229` | none (stack test) | | `check_k2.py --time --no-fse` |

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9 --max-running-requests 64`.
- `--enforce-shared-experts-fusion` is optional. Without it, the split and wide launches and D2 do not run, and K1, K2 and `layer_block` at 16 rows run without the fused shared expert.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_stack https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_stack collector/
```

## Warnings

- The wide launches at 32 rows need `--max-running-requests` of 32 or more and CUDA graphs at 32 rows.
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image. With it, the split and wide launches are not built until they get the Hadamard-32 rotation.
