# collector_261128_qwen_3.8_mono_attn

The image of `collector_261128_qwen_3.8_mono_og` plus two SGLang PRs for the attention front at 32 rows. The plugin is at `3cf33b156d` (`q38v3/20-og`).

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_attn@sha256:8387099e9212e3d841f9f65056d75212693343308c8dc6f207b8fbd49b06a9bf` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | ITT YAMLs [`weekend_1128/attn/`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/weekend_1128/attn), arm `t`. Arm `g` (the og image) is the control. The MLflow parent runs are in `weekend_1128/PLAN.md`. |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `5059e52bd8` | this branch: `102a17d4df` (2026-11-27) plus 2 ports | section "SGLang PRs" |
| AITER | `581579ab51` (2026-11-27) | base image | none |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `3cf33b156d60dc6c86d90e160e84453bd69d7b62` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/20-og), branch `q38v3/20-og` | the plugin of `collector_261128_qwen_3.8_mono_og` |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## SGLang PRs

| Commit | PR | Head | Port | What it does |
| --- | --- | --- | --- | --- |
| `6de48abf91` | [sglang#47014](https://github.com/sgl-project/sglang/pull/47014) | `f010ef9843` | full | The decode KV write also builds the split-KV plan of `vattn_asm`, and `cu_seqlens_q` reuses `qo_indptr_unified_decode`. |
| `5059e52bd8` | [sglang#47779](https://github.com/sgl-project/sglang/pull/47779) (draft), with its base [sglang#45529](https://github.com/sgl-project/sglang/pull/45529) | `83d026e84d` | full | On aiter, the Triton kernel `fused_qk_gemma_rmsnorm_rope_gate` does the Q/K GemmaRMSNorm, the partial RoPE and the gate split in one launch, with one wavefront for each head on HIP. With `copy_gate=False`, `fused_sigmoid_mul` reads the gate in place. |

- The PR base of [sglang#47779](https://github.com/sgl-project/sglang/pull/47779) is `94a4c44813` (2026-11-11). Since then, [sglang#51080](https://github.com/sgl-project/sglang/pull/51080) launches the kernel with PDL. The port keeps `ENABLE_PDL=use_pdl, **pdl_kwargs` and adds `COPY_GATE=copy_gate` and `num_warps=1` on HIP.
- The build applies the diff `102a17d4df..5059e52bd8` to the editable SGLang tree of the base. That keeps the ROCm pyproject files of the nightly. The Dockerfile checks that the diff has the 7 expected files.

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9 --enforce-shared-experts-fusion`, and the run's `--max-running-requests` and CUDA graph sizes.
- NEXTN: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec`.
- `SGLANG_AMD_MONO_K1V=0` turns K1v off.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_attn https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_attn collector/
```

## Warnings

- og and `copy_gate=False`: the gate is a strided view. `_core` of og makes it contiguous, so each attention layer gets one copy kernel. The result stays correct.
- [sglang#47779](https://github.com/sgl-project/sglang/pull/47779) is a draft. Its head can move. This image has the head `83d026e84d`.
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (rotated GDN projections) is not in this image.
