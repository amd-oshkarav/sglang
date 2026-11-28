# collector_261128_qwen_3.8_mono_h32

The 20261127 nightly with [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) (the Hadamard-32 rotation of the online MXFP4 GDN projections) and the Qwen3.8 monokernel stack of sgl-amd-mono with the split and wide launches on the rotated projections, at `5bea396cc3`.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_h32@sha256:f478330fd331431b6f16addea8aa0b042b7724b092f51de580149d430a51d529` |
| Base image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt) |
| Runs | The e2e test of the plugin (the last entry of Crusoe job 332640). The weekend plan is [`weekend_1128/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/weekend_1128). |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `8d612e062a` | this branch: `102a17d4df` (2026-11-27) plus one port commit | [sglang#46059](https://github.com/sgl-project/sglang/pull/46059), section "SGLang" |
| AITER | `581579ab51` (2026-11-27) | base image | none. It has the rotation of `per_1x32_f4_quant_hip` ([aiter#6734](https://github.com/ROCm/aiter/pull/6734), merged as `55645da932`). |
| FlyDSL | 0.4.2 | base image | none |
| sgl-amd-mono | `5bea396cc304312d44d80914c78724c2abf6b680` | [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono/tree/q38v3/15-h32), branch `q38v3/15-h32` | the stack `q38v3/13-stack` (`fb592293c3`) plus one commit, section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git669b31ac.rocm10.0.0 | base image | none |
| ROCm | 10.0.0 | base image | none |

## SGLang

| Commit | PR | PR head used | Port | State (2026-11-28) |
| --- | --- | --- | --- | --- |
| `8d612e062a` | [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) | `122da99f694b0045deba17c085868f3a433d1074` | full | open |

The PR adds `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD` (default on). With it, the online MXFP4 GDN projections (`in_proj_qkvz`, `out_proj`) quantize a Hadamard-32 rotation of the weights and of the activations. The quant state of the layers is `{"hadamard": True, "quant_kwargs": {"hadamard_rotate": True}}`.

## sgl-amd-mono

| Commits | PR | PR head used | State (2026-11-28) |
| --- | --- | --- | --- |
| up to `fb59229` | the stack, see [`collector_261128_qwen_3.8_mono_stack`](https://github.com/amd-oshkarav/sglang/tree/collector_261128_qwen_3.8_mono_stack) | `fb592293c3` | P6 open, the others draft |
| `5bea396` | none yet (branch `q38v3/15-h32`) | `5bea396` | The split launch (front16) and the wide launches at 32 rows run on the rotated projections (`mxfp4_h32`), in place of the rotation guard of the stack. |

## Server

- Environment: `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`.
- Flags: `--enable-amd-monokernel --tp-size 8 --attention-backend aiter --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --page-size 16 --disable-radix-cache --chunked-prefill-size 16384 --mem-fraction-static 0.9 --max-running-requests 64`.
- `--enforce-shared-experts-fusion` is optional. Without it, the split and wide launches and D2 do not run.
- `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0` turns the rotation off. The plugin then runs the unrotated formats, as in the stack image.

## Build again

```bash
git clone --depth 1 --branch collector_261128_qwen_3.8_mono_h32 https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_h32 collector/
```

## Warnings

- The Dockerfile checks out SGLang `8d612e062a` over the SGLang of the base image (`/sgl-workspace/sglang`, an editable install). The build checks that `sglang` imports from that tree and that the PR's environment variable exists.
- [sglang#46059](https://github.com/sgl-project/sglang/pull/46059) is open. A new PR head can change the quant state that the plugin reads.
- The wide launches at 32 rows need `--max-running-requests` of 32 or more and CUDA graphs at 32 rows.
