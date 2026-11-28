# collector_261128_qwen_3.8_mono_all

The 20261127 nightly with all the Qwen3.8 monokernel work of the weekend of 2026-11-28 at sgl-amd-mono `e65acb91c3`, with no SGLang change. The plugin is the stack plus item 1 of vllm-project/vllm#61271, K1v, Hadamard-32 in the split and wide launches, and the no-FSE support.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_all@sha256:42ab51f07deba9ce622f0e0be8c2f88c72fb66c414682479b3d5b865adf28d7f` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/allfse/` and `weekend_1128/allnofse/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/weekend_1128), arm `a`. The MLflow parent runs are in `weekend_1128/PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) | base image | none |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `e65acb91c31d2b891f5f06f5992ab739f9be7789` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/19-all), branch `q38v3/19-all` | the stack plus 8 commits, section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## sgl-amd-mono

The branch is `fb59229` (the stack) with `git cherry-pick -x` of the commits below. Each commit message names its source commit.

| Commits | PR | Source branch | What it does |
| --- | --- | --- | --- |
| up to `fb59229` | the stack: [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) (P6), [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159), [sgl-amd-mono#174](https://github.com/sgl-project/sgl-amd-mono/pull/174), [sgl-amd-mono#180](https://github.com/sgl-project/sgl-amd-mono/pull/180), the rotation guard and D2 | `q38v3/13-stack` | open and draft |
| from `2c86e6c`, `2fdd582` | none yet (item 1) | `q38v3/14-up` | `layer_block` gives each expert pick of 2 and 3 rows its own slot (item 1 of vllm-project/vllm#61271), `UP_MAX = 3` |
| from `d40d683`, `c46216b` | none yet (K1v) | `q38v3/16-k1v` | K1 runs the GDN front of the NEXTN verify steps of up to 8 rows on the ReplaySSM ring of SGLang |
| from `5bea396` | none yet (Hadamard-32) | `q38v3/15-h32` | the split and wide launches read a GDN projection that [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) rotated |
| from `adabdda`, `2b8c598`, `23fe99e` | none yet (no-FSE) | `q38v3/18-nofse` | the split, wide and `layer_block` launches without `--enforce-shared-experts-fusion` |

The merge of the cherry-picks changed 3 places by hand: `_builds_split` and `_builds_wide` in `spec.py`, the `_check` helper of the e2e test, and the format and FSE loops of `check_wide32.py`. `weekend_1128/PLAN.md`, section 7, gives the details.

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9`, and the run's `--max-running-requests` and CUDA graph sizes.
- NEXTN: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec`. K1v needs the ReplaySSM ring.
- The image runs with and without `--enforce-shared-experts-fusion`. The `allfse` arms set the flag, the `allnofse` arms do not.
- `SGLANG_AMD_MONO_K1V=0` turns K1v off.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_all https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_all collector/
```

## Warnings

- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image. The Hadamard-32 path runs only in the kernel tests.
- K1v runs only in verify steps of 1 to 8 rows: 1 and 2 requests with 4 draft tokens.
