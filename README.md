# Qwen3.8 MXFP4 route rows: the pristine nightly with AITER main

Image: `qwen38-pristine-mxfp4:4120f9839`. The image is local only. The registry does not have it. Job 313912 built it on `crsuse2-m2m-v3-218` (2026-11-26, 10:58 to 11:56 UTC). The build log gives the manifest list `sha256:0097d6a00ab9b8d9c60fa5bb0b8bc78680da6b5dad85cfe9dc5870b740283923`, triton 3.9.0 and flydsl 0.5.3. The build log is `~/oob/logs/qwen38-mxfp4/build_313912.log` on the Crusoe login.

This branch is SGLang main `512127045d` (2026-11-26) with no code change. That commit is the SGLang of the nightly image `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261126`. The only commit of the branch is this README.

The image moves AITER to main `4120f9839`. That AITER has the MXFP4 route-row switch of [aiter#7706](https://github.com/ROCm/aiter/pull/7706). The image also has one AITER row edit. The section "AITER" gives it.

The experiment is [`qwen38/spec_decode_mi355/p1b_mxfp4_rows/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/olga/qwen38/spec_decode_mi355/p1b_mxfp4_rows/PLAN.md) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`.

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-26) |
| --- | --- | --- | --- |
| - | No SGLang change. The image runs `512127045d` as the nightly gives it. | - | - |

## AITER

The image uses AITER main [`4120f9839`](https://github.com/ROCm/aiter/commit/4120f98392c0f4437b572b5f131a62444c125435) (2026-11-26). The pin of the nightly image is `581579ab5`.

| Pull request | Commit | Use in this image |
| --- | --- | --- |
| [aiter#7706](https://github.com/ROCm/aiter/pull/7706) | `77ea506d9` | `AITER_MXFP4_INTERMEDIATE=1`: stage 2 writes MXFP4 route rows (e2m1 values and one e8m0 scale for each 32 values) |
| [aiter#8669](https://github.com/ROCm/aiter/pull/8669) | `04500613c` | The FP8 row table for `AITER_FLYDSL_STAGE2_FP8=1` |

| Edit | AITER file | Change | Source file | On AITER main |
| --- | --- | --- | --- | --- |
| FlyDSL rows for both switches | `aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv` | Replaces the 3 Q256 rows (E 513 at 8192, 16384 and 32768 tokens) with the 3 FlyDSL reduce rows of `qwen3_8_2400b_fp4_tuned_stage2_fp8_fmoe.csv` of the same commit. The other rows do not change. | The second `RUN` of [`p1b_mxfp4_rows/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/Dockerfile) | No. With FP8 off, main gives the Q256 rows. |

- triton is 3.9.0 (`.github/scripts/install_triton.sh` of AITER). The nightly has 3.8.
- FlyDSL is 0.5.x (`requirements.txt` of AITER).
- torch, torchvision and torchaudio stay at the versions of the nightly. The Dockerfile checks it.
- The kernels are prebuilt: `PREBUILD_KERNELS=1`, `GPU_ARCHS=gfx950`.

**Warning:** FP8 wins when both switches are set. Set `AITER_FLYDSL_STAGE2_FP8=0` with `AITER_MXFP4_INTERMEDIATE=1`.

**Warning:** without the row edit, AITER main gives the Q256 rows of [aiter#5988](https://github.com/ROCm/aiter/pull/5988) at 8192 to 32768 tokens when FP8 is off. Q256 ignores both switches. Then `AITER_MXFP4_INTERMEDIATE=1` has no effect at those token counts.

## sgl-amd-mono

`python/pyproject_other.toml` of `512127045d` pins sgl-amd-mono `32a122031b`. The arms do not set `--enable-amd-monokernel`, so the plugin does not run.

## Image

- Image: `qwen38-pristine-mxfp4:4120f9839`, local only.
- Base image: `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261126@sha256:98b812618fd9b36ec23db015f24c3f3559adef9d2f71426a124f442d57d574b1`.
- Start commit of the branch: `512127045d`.

[`p1b_mxfp4_rows/gate.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/gate.sh) builds the image on the node. The file links are at commit [`664a3683`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/664a3683324393eab7378cea3c4bf23f7161b23b) of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`.

1. The first `RUN` of [`Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/Dockerfile) moves AITER in `/sgl-workspace/aiter` from `581579ab5` to `4120f9839`.
   - It installs the AITER requirements with torch fixed, installs triton 3.9.0, prebuilds the kernels and installs AITER in editable mode.
   - It checks the SGLang commit, torch, FlyDSL 0.5, the count of the JIT modules and the MXFP4 switch.
2. The second `RUN` does the row edit (section "AITER"). It writes `/sgl-workspace/mxfp4_rows.txt`.
3. `gate.sh` imports the server, the GDN backend and the Qwen3.8 model files on GPU 0.

No layer comes from an ITT `dockerfile` key. To build the image from a checkout of `664a3683`:

```bash
docker build -t qwen38-pristine-mxfp4:4120f9839 - < qwen38/spec_decode_mi355/p1b_mxfp4_rows/Dockerfile
```

## Server

The arm YAMLs are [`fp8.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/fp8.yaml), [`fp4.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/fp4.yaml) and [`fp4_evals.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/fp4_evals.yaml). [`gen_yamls.py`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/664a3683324393eab7378cea3c4bf23f7161b23b/qwen38/spec_decode_mi355/p1b_mxfp4_rows/gen_yamls.py) writes them.

| Arm | `AITER_FLYDSL_STAGE2_FP8` | `AITER_MXFP4_INTERMEDIATE` |
| --- | --- | --- |
| A, FP8 route rows | 1 | 0 |
| B, MXFP4 route rows | 0 | 1 |

- Environment of both arms: `AITER_LOG_TUNED_CONFIG=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `SAFETENSORS_FAST_GPU=1`, `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1`, `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`.
- Flags of both arms: `--tp-size 8 --attention-backend aiter --chunked-prefill-size 16384 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --enforce-shared-experts-fusion --disable-radix-cache --mem-fraction-static 0.85 --max-running-requests 256 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder`.

## Results

- Op bench (m15-33, AITER `4120f9839`): with MXFP4 rows, `fused_moe` is 12.6% to 14.4% faster than with FP8 rows at 8192 to 32768 tokens. [MLflow](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/d3cd2de1377141c2b1a9c1b03d9a04a7).
- Serving A/B (Crusoe job 313912, `crsuse2-m2m-v3-218`, order A B B A, MTP NEXTN 3/1/4): MXFP4 rows (B) against FP8 rows (A) give output throughput +1.8% and +1.9%, mean TTFT -3.7% and -3.9%, mean TPOT -1.7% and -1.6%. These are the geomeans of 14 cells. The drift between the rounds is 0.3% or less.
- Evals (A / B): chat GSM8K flexible-extract 0.9773 / 0.9795, GPQA Diamond 3 runs 0.921 / 0.929, NIAH 64k 1.000 / 0.998, NIAH 128k 1.000 / 0.998. All differences are in the noise.
- MLflow parent runs: [fp8](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/78f7e3e961da4eb5a25c74f2f3be0250), [fp4](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/a464eb98caaa4114bd13fa5554e96b88), [fp4_r2](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/064e2cec652742518ffb2885407f6fd8), [fp8_r2](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/ed801cf810704a7781fd1be4f7bf3ddc), [fp4_evals](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/717c1469cc0243f685588de064899091), [fp8_evals](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/4204262422411969/runs/af50a8057d6b4f399d69b3005d322e3d).
- The full tables: [PLAN.md](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/9f371768/qwen38/spec_decode_mi355/p1b_mxfp4_rows/PLAN.md), section "Phase 2 result".
