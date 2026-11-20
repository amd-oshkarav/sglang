# week_47 replay

MLflow run [11fa02620a5b480c8031781310d0a241](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/11fa02620a5b480c8031781310d0a241?o=4032599967441609), name `20261119_sglang_tp8_fp4_collector_09_11_3590bb91_mono154_410ca78_gpqa_mi355x`. This is the replay of week_47, run [718975bdf2e44f91b676dd526d45fa3b](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/718975bdf2e44f91b676dd526d45fa3b?o=4032599967441609).

The sweep config is `docs/examples/tonight/qwen_3.8_collector_09_11_stack_p2_mono154.yaml`. It runs all twelve `sglang_bench_serve` cells of the week_47 sweep, then GPQA (`sglang_run_eval`, 32 threads, `max-tokens` 100000, temperature 1.0, top-p 0.95, thinking enabled). GPQA scored 0.9141 on the 8192/1024 concurrency-4 child. The host was `crsuse2-m2m-v3-086` (EPYC 9575F), ITT 1.86.0.

| ISL | OSL | Conc | Prompts |
| --- | --- | --- | --- |
| 8192 | 1024 | 4, 8, 16, 32, 64, 128, 256 | 32, 64, 96, 128, 256, 384, 768 |
| 60000 | 600 | 4, 16, 64 | 32, 96, 256 |
| 128000 | 1024 | 4, 16 | 32, 96 |

The server uses `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. Page size is 16. There is no MTP.

## Image

`amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_09_11_3590bb91_mono154_410ca78`, digest `sha256:016361d14a62a5cc22c94414834d0f0fe8a6218dfa093d2c38ffb55377dafdb7`, built 2026-11-19T15:50:51Z.

The YAML builds it from [`docker/collector_09_11_qwen_3.8_stack_p2_mono154/Dockerfile`](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/f18453209ee14ba690fd979d45775d57/artifacts?selectedArtifact=f18453209ee14ba690fd979d45775d57%3Acollector_09_11_qwen_3.8_stack_p2_mono154%2FDockerfile&o=4032599967441609) (`dockerfile.path` `/app/repos/inference-testing/docker/collector_09_11_qwen_3.8_stack_p2_mono154/Dockerfile`), logged on child run `f1845320` of the week_47 replay. That file is one layer on the base image below. It does not rebuild SGLang or AITER.

1. Apply `register_model_type.patch`, which is commit `5cdfd1933b9726df3c7e79a9e55fb339e875ead9` (#46036), so `register()` takes `model_type`.
2. Uninstall the base image's monokernel and install [omirosh/sgl-amd-mono](https://github.com/omirosh/sgl-amd-mono) at `410ca7872d8029a5532c39fe1930d9a1b23ac77a`, the 2026-11-18 head of [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154). The later head `05b9dd9f` calls `state_layer_cache`, which this SGLang tree does not have.

## Base image

`amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_18_11_on_09_11_3590bb91_aiter_90596736_mono_pr92_pr147`, built by [`docker/collector_09_11_qwen_3.8_stack_p2/Dockerfile`](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/cc150acd4a624b5d9a81af377fd3747b/artifacts?selectedArtifact=cc150acd4a624b5d9a81af377fd3747b%3Acollector_09_11_qwen_3.8_stack_p2%2FDockerfile&o=4032599967441609). That Dockerfile is logged on the finished run `cc150acd` (`test_20261118_sglang_tp8_fp4_collector_18_11_on_09_11_3590bb91_aiter_90596736_mi355x`), which builds the same base without the mono layer. The `_mono_pr92_pr147` run `5f417bc9` is still running and has no artifacts yet. It starts from the Q1 image `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_26_qr_flydsl_int4_mono_plugin` (`sha256:8b65d34aba0f7c257def5aaf6830646339ae42c6dd6eba2bef997bd884f46bd0`), whose SGLang is `493cd5e719`.

1. Replace AITER with `90596736129ccc0bcf720686cf202067a11f110d` (2026-10-26). Apply the P-1b edit in `aiter/ops/flydsl/moe_kernels.py`, changing `if force_reduce or fp8_reduce:` to `if force_reduce:`. Copy the two a4w4 GEMM tables from [aiter#7441](https://github.com/ROCm/aiter/pull/7441) (`ec91f61f0c`). Reinstall AITER.
2. Apply `stack_on_493.diff`, the code diff from `493cd5e719` to `3590bb919ddb321b61b2075b5ed2b37fc67a4b74`. That diff carries #47219, #47433, #45529, #41683, #47269, #45555 and #39595. Kernels are not rebuilt.
3. Replace the monokernel with sgl-project `main` `b504de5c`, then cherry-pick PR 92 (`5516f207`, `262dd241`, `89753935`, `a4c94bb9`) and PR 147 (`181467fe`). The extra layer above uninstalls this plugin and installs `410ca7872d`.

## Pins

| | week_47 | this replay |
| --- | --- | --- |
| Image | `qwen38-collector:1811-stack-p2-mono154` (`sha256:f90b4986`) | `..._collector_09_11_3590bb91_mono154_410ca78` (`sha256:016361d1`) |
| SGLang | `afea911d7b` on `collector_18_11_qwen_3.8_stack_p2_mono154` | `3590bb919d` on `493cd5e719`, plus register commit `5cdfd1933b` |
| AITER | `90596736`, a4w4 tables, `aiter_patch` none | `90596736`, a4w4 tables from `ec91f61f`, P-1b `force_reduce-only` |
| Monokernel | `410ca7872d` | `omirosh/sgl-amd-mono` `410ca7872d` |
| Host | `crsuse2-m2m-v3-252` | `crsuse2-m2m-v3-086` |
