# collector_261127_qwen_3.8_best_aiter8829

The week 47 best non-MTP stack of Qwen3.8 on MI355X, plus [aiter#8829](https://github.com/ROCm/aiter/pull/8829): the FlyDSL stage 2 rows also apply with `AITER_MXFP4_INTERMEDIATE=1`.

| Item | Value |
| --- | --- |
| Image | `amdsiloai/sglang-private:collector_261127_qwen_3.8_best_aiter8829@sha256:9c52379874ae345087d4f7ec004f1b65e91c2d03014868331c1ac867956f65ff` |
| Base image | `lmsysorg/sglang:v0.5.23-rocm10-mi35x@sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5` |
| Build files | [`collector/`](collector/): [`Dockerfile`](collector/Dockerfile), [`pip_freeze.txt`](collector/pip_freeze.txt), [`aiter_setup.sh`](collector/aiter_setup.sh), [`aiter_pr8829.patch`](collector/aiter_pr8829.patch) |
| Runs | MLflow parent: added after the run. ITT YAML [`full.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/1f11f1441ccfa19e4468081691bd9533ed74cfd9/qwen38/spec_decode_mi355/best_261127_aiter8829/full.yaml) |

## Versions

| Part | Version | Source | Change against the base image |
| --- | --- | --- | --- |
| SGLang | `271511b7fa` (this branch) on main `88dc7501f5` (2026-11-24) | this branch | 6 commits, section "SGLang" |
| sgl-kernel | `sglang-kernel` 0.4.10, built from `271511b7fa` | this branch | rebuilt |
| AITER | `4120f98392` (2026-11-26) plus one patch | ROCm/aiter | 1 patch, section "AITER" |
| FlyDSL | 0.5.3 | `requirements.txt` of AITER | |
| sgl-amd-mono | `4cba5b253e` plus 5 commits of PR 154, cherry-picked in the build (HEAD `021190585e` in this image) | sgl-project/sgl-amd-mono | section "sgl-amd-mono" |
| torch | 2.11.0+rocm10.0.0 | base image | none |
| Triton | 3.8.0+git4cff872c.rocm10.0.0 | base image (`AITER_USE_SYSTEM_TRITON=1`) | none |
| ROCm | 10.0.0 | base image | none |

## SGLang

The commits are the commits of [collector_24_11](https://github.com/amd-oshkarav/sglang/tree/collector_24_11), with the same patch IDs.

| Commit | PR | Picked from collector_24_11 | Change | Port | State on main (2026-11-27) |
| --- | --- | --- | --- | --- | --- |
| `7ae8c9949b` | [sglang#40771](https://github.com/sgl-project/sglang/pull/40771) | `665ddf2c74` | Prefer QuickReduce before custom all-reduce | full | open |
| `e0c745a0e4` | none, commit [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | `3e49b12043` | Use FlyDSL INT4 QuickReduce for contiguous bf16 | full | not merged; [sglang#46850](https://github.com/sgl-project/sglang/pull/46850) is open for the same feature |
| `27190bb389` | [sglang#47219](https://github.com/sgl-project/sglang/pull/47219) | `5ef3f38228` | Fuse add-RMSNorm into FlyDSL INT4 QuickReduce | full | closed by the stale bot, not merged |
| `9225da559a` | [sglang#47433](https://github.com/sgl-project/sglang/pull/47433) | `7e8d8d53ae` | Use one ReplaySSM pipeline stage on gfx950 | full | open |
| `c4bd0585f0` | [sglang#45529](https://github.com/sgl-project/sglang/pull/45529) | `4f1547fd70` | Fuse self-attn prepare on gfx950 (norm, RoPE, gate) | full | open |
| `271511b7fa` | [sglang#39595](https://github.com/sgl-project/sglang/pull/39595) | `4af7130f0d` | Add the FlyDSL GDN prefill backend | full | merged 2026-11-27, after the base |

Not ported: [sglang#39069](https://github.com/sgl-project/sglang/pull/39069), the draft path for bundled MTP. The PR is closed, and a non-MTP run does not use it. The build stops if its code is in the tree.

## AITER

| Edit | Files | Patch | On AITER main |
| --- | --- | --- | --- |
| [aiter#8829](https://github.com/ROCm/aiter/pull/8829) at `688913c38d`: the stage 2 FP8 rows also apply with `AITER_MXFP4_INTERMEDIATE=1` | `aiter/fused_moe.py`, `aiter/aot/flydsl/moe.py`, `op_tests/test_moe_stage2_fp8_rows.py` | [`collector/aiter_pr8829.patch`](collector/aiter_pr8829.patch) | draft PR, not merged |

## sgl-amd-mono

| Commit | PR | PR head used | State (2026-11-27) |
| --- | --- | --- | --- |
| 5 commits on `4cba5b253e` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154) | `05b9dd9fab` | open; the PR head is now `d28e8399dc` |

## Server

- Environment: `AITER_FLYDSL_STAGE2_FP8=0`, `AITER_MXFP4_INTERMEDIATE=1`, `AITER_LOG_TUNED_CONFIG=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `SAFETENSORS_FAST_GPU=1`, `SGLANG_ROCM_GDN_ONLINE_MXFP4=1`, `SGLANG_ROCM_USE_AITER_TOPK_GATING=1`, `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `SGLANG_WARMUP_TIMEOUT=36000`
- Flags: `--tp-size 8`, `--attention-backend aiter`, `--kv-cache-dtype fp8_e4m3`, `--linear-attn-prefill-backend flydsl`, `--mamba-ssm-dtype bfloat16`, `--chunked-prefill-size 16384`, `--max-prefill-tokens 16384`, `--max-running-requests 256`, `--mem-fraction-static 0.9`, `--page-size 16`, `--disable-radix-cache`, `--enable-aiter-allreduce-fusion`, `--enable-amd-monokernel`, `--enforce-shared-experts-fusion`, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `--watchdog-timeout 24000`

## Build again

```bash
git clone --depth 1 --branch collector_261127_qwen_3.8_best_aiter8829 https://github.com/amd-oshkarav/sglang.git && cd sglang
docker build --network=host -t amdsiloai/sglang-private:collector_261127_qwen_3.8_best_aiter8829 collector/
```

## Warnings

- `AITER_FLYDSL_STAGE2_FP8=1` wins over `AITER_MXFP4_INTERMEDIATE=1`. Set `AITER_FLYDSL_STAGE2_FP8=0` for the MXFP4 arm.
- An MTP run on this image must set `--speculative-draft-model-path`, because [sglang#39069](https://github.com/sgl-project/sglang/pull/39069) is not ported.
