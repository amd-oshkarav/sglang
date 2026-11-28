# collector_261128_qwen_3.8_mono_all_pr47014

The image of `collector_261128_qwen_3.8_mono_all` (the plugin at sgl-amd-mono `e65acb91c3`) plus [sglang#47014](https://github.com/sgl-project/sglang/pull/47014). The decode KV write also builds the split-KV plan of `vattn_asm`, and `cu_seqlens_q` reuses `qo_indptr_unified_decode`. Thus each full-attention layer runs no `torch.arange` kernel and no `_vattn_seg_plan_kernel`.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_all_pr47014@sha256:0405ff9a3fc9a1b969b1ddb3a05580ad0996840d42faab8963f8fe21b5286cca` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/allfse/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/weekend_1128), arm `x`. The MLflow parent runs are in `weekend_1128/PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `102a17d4df` (2026-11-27) plus `6de48abf91` | base image, and the [Port] commit of this branch | [sglang#47014](https://github.com/sgl-project/sglang/pull/47014), section "SGLang" |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `e65acb91c31d2b891f5f06f5992ab739f9be7789` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/19-all), branch `q38v3/19-all` | the plugin of `collector_261128_qwen_3.8_mono_all` |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

`pip_freeze.txt` gives `sglang==0.5.23.dev20261127+g102a17d4df`, because the build applies the port to the editable tree of the base and does not install SGLang again.

## SGLang

| Commit | PR | PR head | Port |
| --- | --- | --- | --- |
| `6de48abf91` | [sglang#47014](https://github.com/sgl-project/sglang/pull/47014) (open) | `f010ef98431a1f6ce85a565ca6027fd696dec742` | full, 4 files |

The build applies the diff `102a17d4df..6de48abf91` to `/sgl-workspace/sglang` of the base image. Thus the ROCm `pyproject` files of the nightly build stay. The build then checks that the 4 files are the files of `6de48abf91`, and imports `aiter_backend` from `/`.

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9`, and the run's `--max-running-requests` and CUDA graph sizes.
- The image runs with and without `--enforce-shared-experts-fusion`.
- `SGLANG_AMD_MONO_K1V=0` turns K1v off.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_all_pr47014 https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_all_pr47014 collector/
```

## Warnings

- In `/sgl-workspace`, `import sglang` finds a namespace package, and `sglang.__file__` is None. The build imports a real module from `/` (`966e125af3`).
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image.
