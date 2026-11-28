# collector_261128_qwen_3.8_mono_k1v

The 20261127 nightly with the Qwen3.8 monokernel stack of sgl-amd-mono plus K1v at `c46216b72e`, with no SGLang change.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_k1v@sha256:fc4b501cab5917a277bf21d622a7009063cffb928e59e04ba725df77ce0f28fc` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/k1v/` and `weekend_1128/k1vnofse/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/77085eed/qwen38/spec_decode_mi355/mono_plugin/weekend_1128), arms `v` (K1v on) and `w` (K1v off). The MLflow parent runs are in its `PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) | base image | none |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `c46216b72e36961b46157e5d06961b3de418b730` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/16-k1v), branch `q38v3/16-k1v` | the stack plus 2 commits, section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## sgl-amd-mono

| Commits | PR | PR head used | State (2026-11-28) |
| --- | --- | --- | --- |
| up to `fb59229` | the stack: [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) (P6), [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159), [sgl-amd-mono#174](https://github.com/sgl-project/sgl-amd-mono/pull/174), [sgl-amd-mono#180](https://github.com/sgl-project/sgl-amd-mono/pull/180), the rotation guard and D2 | the heads of [`collector_261128_qwen_3.8_mono_stack`](https://github.com/amd-oshkarav/sglang/tree/collector_261128_qwen_3.8_mono_stack) | open and draft |
| `d40d683` | none yet (K1v) | | K1 runs the GDN front of the NEXTN verify steps of up to 8 rows on the ReplaySSM ring of SGLang |
| `c46216b` | none yet (K1v test) | | the e2e test checks K1v in check mode and on the verify server |

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- `SGLANG_AMD_MONO_K1V=0` turns K1v off. Then the image runs the code of the stack.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9 --max-running-requests 16 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec`.
- `--enforce-shared-experts-fusion` is optional. K1v is the GDN front, so it runs in both modes.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_k1v https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_k1v collector/
```

## Warnings

- K1v needs `--enable-linear-replayssm-spec`. Without the ReplaySSM ring, the verify steps run the GDN front of SGLang.
- K1v runs only in verify steps of 1 to 8 rows: 1 and 2 requests with 4 draft tokens. Wider steps run the code of the stack.
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image.
