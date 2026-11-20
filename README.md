# collector_19_11

Branch `collector_19_11` starts from SGLang main [`b1970bc473`](https://github.com/sgl-project/sglang/commit/b1970bc473cfff1731c068e7d23e9e61b794b0c0) (`b1970bc473cfff1731c068e7d23e9e61b794b0c0`), 2026-11-19 00:52 -0800:

`[Feat] cli: Export Cookbook commands for every config-driven LLM/VLM page (#50106)`

Each row below is one commit on top of that main. The source list is the SGLang table on [`collector_09_11_qwen_3.8_stack_p2`](https://github.com/amd-oshkarav/sglang/tree/collector_09_11_qwen_3.8_stack_p2). Merge status was checked against `sgl-project/sglang` main, not against that table.

## Ported features

| Commit | Pull request | Feature |
| --- | --- | --- |
| `7502002d4c` | [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP |
| `cea365f745` | [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce |
| `d1b489d47f` | [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | Use FlyDSL INT4 QuickReduce for contiguous bf16 |
| `d10534a597` | [#47219](https://github.com/sgl-project/sglang/pull/47219) | Fuse add-RMSNorm into FlyDSL INT4 QuickReduce |
| `af1219fd5e` | [#47433](https://github.com/sgl-project/sglang/pull/47433) | Use one ReplaySSM pipeline stage on gfx950 |
| `bc161b2bb6` | [#45529](https://github.com/sgl-project/sglang/pull/45529) | Fuse self-attn prepare on gfx950 (norm, RoPE, gate) |
| `d9313039d0` | [#39595](https://github.com/sgl-project/sglang/pull/39595) | Add the FlyDSL GDN prefill backend |

#40771 and #47219 were ported as those features. Their open pull requests also contain other files, which were not brought over.

## Model and sweep

Qwen3.8-2.4T TP8 FP4 on MI355X. This branch is SGLang `f3371c01c2` (`collector_19_11`). The image that serves it is the collector image plus one layer that installs [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154).

AITER is `90596736129ccc0bcf720686cf202067a11f110d` (2026-10-26). That commit is before [aiter#5988](https://github.com/ROCm/aiter/pull/5988) and does not contain [aiter#7441](https://github.com/ROCm/aiter/pull/7441). The image adds that pull request's two GEMM tables from `ec91f61f0c550811cf121f0b0dc667c09524b161` and does not apply the P-1b edit.

The monokernel is [sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) `main` at `32a122031b7b63858c1610370b49a3bf9a215997`, plus the five commits of PR 154 through `05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68`.

## Benchmark

The sweep is `docs/examples/tonight/qwen_3.8_collector_19_11_mono.yaml` in `AMD-AGI/inference-testing`. It starts the mono image below and runs `sglang_bench_serve` on random data (`random-range-ratio` 0.8, `request-rate` inf). All twelve cells run:

| ISL | OSL | Conc | Prompts |
| --- | --- | --- | --- |
| 8192 | 1024 | 4, 8, 16, 32, 64, 128, 256 | 32, 64, 96, 128, 256, 384, 768 |
| 60000 | 600 | 4, 16, 64 | 32, 96, 256 |
| 128000 | 1024 | 4, 16 | 32, 96 |

GPQA follows the sweep (`sglang_run_eval`, 32 threads, `max-tokens` 100000, temperature 1.0, top-p 0.95, thinking enabled). RULER `niah_single_2` is in the file and commented out. The finished run is [8ddc5f7ab5f545ee8ea7df576191e8d4](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/8ddc5f7ab5f545ee8ea7df576191e8d4?o=4032599967441609).

The server uses `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. Page size is 16, unified attention and GDN online MXFP4 are on, and the prefill chunk is 16384 tokens.

## Images

The serving image is `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736_mono_32a12203_pr154`. It is the base image plus the sgl-amd-mono layer.

### sgl-amd-mono#154 layer

Dockerfile: [`collector_19_11_mono/Dockerfile`](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/5d8e093471b5417aa9d0d8446e92efbf/artifacts?selectedArtifact=5d8e093471b5417aa9d0d8446e92efbf%3Acollector_19_11_mono%2FDockerfile&o=4032599967441609), logged on child run `5d8e0934` of MLflow run `8ddc5f7a`. The sweep config that builds this tag is `docs/examples/tonight/qwen_3.8_collector_19_11_mono.yaml`, and its `dockerfile.path` is `/app/repos/inference-testing/docker/collector_19_11_mono/Dockerfile`.

This file starts `FROM` `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736` and leaves SGLang and AITER as that image built them. It installs the monokernel plugin:

1. Uninstall any `sgl-amd-mono` already present and clone [sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) `main`.
2. Detach at `32a122031b7b63858c1610370b49a3bf9a215997` and refuse any other commit.
3. Check that SGLang's `register()` already accepts `model_type`. PR 154 calls it that way.
4. Fetch `pull/154/head` and cherry-pick these five commits, in order, onto that main pin. The pull request is open, and its base is one commit behind main, so the image records the main pin and the cherry-picks rather than a single remote SHA.
   - `7fc7b73e6f40e52fd6a9eada64508b148d5ba71c` runs the tail of a 16-row step as one `layer_block` launch.
   - `289c1a2d611e085db6a2b83859b1f4db614e4e9f`, `1ec6f748a8251d572fafac22e38e257684b271d5`, `34d8f8cbb88050788e62e880601cca08838ef5ac` and `05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68` are the tests for that launch. `05b9dd9f` is the head.
5. Require that main is an ancestor and that exactly those five commits sit on top of it.
6. `pip install --no-deps -e .` and require the `sgl_amd_mono` entry point under `sglang.srt.plugins`.

`layer_block` runs when the server is started with `--enable-amd-monokernel` and `--enforce-shared-experts-fusion`.

### Base image

The tag is `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736`.

`docker/collector_19_11/Dockerfile` and `docker/collector_19_11/aiter_7441.patch` in `AMD-AGI/inference-testing` build this tag. No MLflow run was logged for this pin, so both files are copied below. The only other logged `collector_19_11/Dockerfile` is on run [a0908c0b](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/a0908c0bb697401ba44359fa133476ec/artifacts?selectedArtifact=a0908c0bb697401ba44359fa133476ec%3Acollector_19_11%2FDockerfile&o=4032599967441609), and that one builds AITER `8f4fe9e5`. The config that builds this tag is `docs/examples/collector/qwen_3.8_collector_19_11.yaml`, and its `dockerfile.path` is `/app/repos/inference-testing/docker/collector_19_11/Dockerfile`.

It starts from `lmsysorg/sglang:v0.5.23-rocm10-mi35x` and builds for gfx950:

1. Install Rust if the base image has no `cargo`.
2. Clone [amd-oshkarav/sglang](https://github.com/amd-oshkarav/sglang) branch `collector_19_11` and detach at `f3371c01c2f9292a0f4e44ed2de212959c0342f6`, with submodules. The checkout is used as it is. On top of SGLang main `b1970bc473` that commit carries #39069 (bundled MTP draft path), #40771 (QuickReduce before custom all-reduce), FlyDSL INT4 QuickReduce (`8cb39392`), #47219 (add-RMSNorm in that QuickReduce), #47433 (one ReplaySSM stage on gfx950), #45529 (fused self-attention prepare) and #39595 (FlyDSL GDN prefill).
3. Build the ROCm kernels from `sgl-kernel` or `python/sglang/kernels/aot`.
4. Install that checkout in editable mode. The extra is `all_hip` when the base torch is 2.9 or 2.10, and `all_hip_rocm724` otherwise. Torch and Triton stay pinned to the base image, `sgl-eval==0.1.2` is installed, and the build stops if a CUDA wheel was pulled in.
5. Replace AITER. Clone [ROCm/aiter](https://github.com/ROCm/aiter) `main` and detach at `90596736129ccc0bcf720686cf202067a11f110d` (2026-10-26). That commit is 228 commits before `8f4fe9e5` and is before [aiter#5988](https://github.com/ROCm/aiter/pull/5988). It does not contain [aiter#7441](https://github.com/ROCm/aiter/pull/7441). Apply `aiter_7441.patch`, which is that pull request (`ec91f61f0c`) and adds the two Qwen3.8 TP8 a4w4 GEMM tables (`qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv` and the untuned table, N 4608, K 8192). Reinstall AITER from that tree and import `QuickAllReduceInt4`. The P-1b one-line edit is not applied.
6. Check again that SGLang is still `f3371c01` and that the FlyDSL GDN prefill backend is present.


`docker/collector_19_11/Dockerfile`:

```dockerfile
# Collector image. Clones amd-oshkarav/sglang collector_19_11 at
# SGLANG_REV and reinstalls that checkout. Nothing in the SGLang tree
# is patched.
#
# AITER is 90596736 (2026-10-26). That base does not contain aiter#7441
# or aiter#5988. The only change applied to it is aiter_7441.patch, the
# two Qwen3.8 TP8 a4w4 GEMM tables from ec91f61f0c. AITER is then
# reinstalled. 90596736 is 228 commits before 8f4fe9e5, which is the
# parent of aiter#5988.
#
#   docker build -f docker/collector_19_11/Dockerfile \
#       --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x \
#       -t amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736 \
#       docker/collector_19_11
#
# ITT needs dockerfile.path to be this file:
#   /app/repos/inference-testing/docker/collector_19_11/Dockerfile

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG SGLANG_REPO=https://github.com/amd-oshkarav/sglang.git
ARG SGLANG_BRANCH=collector_19_11
ARG SGLANG_DIR=/sgl-workspace/sglang
ARG SGLANG_REV=f3371c01c2f9292a0f4e44ed2de212959c0342f6
ARG SGLANG_EXTRAS=auto
ARG SGL_EVAL_VERSION=0.1.2
ARG AITER_DIR=/sgl-workspace/aiter
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REV=90596736129ccc0bcf720686cf202067a11f110d

ENV GPU_ARCHS=gfx950
ENV PYTORCH_ROCM_ARCH=gfx950
ENV AMDGPU_TARGETS=gfx950
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_BUILD_JOBS=4

WORKDIR /sgl-workspace

RUN set -eu; \
    if command -v cargo >/dev/null 2>&1; then \
        echo "cargo already present: $(cargo --version)"; \
    else \
        curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 \
            --retry-all-errors -sSf https://sh.rustup.rs | sh -s -- -y; \
        rustc --version; \
        cargo --version; \
    fi

# Blobless rather than shallow: setuptools-scm reads tags and the distance to
# them, and --depth 1 would install the package as 0.0.0.dev0.
RUN set -eu; \
    git config --global --add safe.directory '*'; \
    python3 -m pip uninstall -y sglang sgl-kernel sgl_kernel 2>/dev/null || true; \
    rm -rf "${SGLANG_DIR}"; \
    git clone --filter=blob:none --branch "${SGLANG_BRANCH}" \
        "${SGLANG_REPO}" "${SGLANG_DIR}"; \
    cd "${SGLANG_DIR}"; \
    git checkout --detach "${SGLANG_REV}"; \
    got="$(git rev-parse HEAD)"; \
    case "${got}" in \
        "${SGLANG_REV}"*) ;; \
        *) echo "ERROR: asked for ${SGLANG_REV}, got ${got}" >&2; exit 1 ;; \
    esac; \
    git submodule sync --recursive; \
    git submodule update --init --recursive; \
    echo "sglang at ${SGLANG_BRANCH}, HEAD $(git rev-parse HEAD)"

RUN set -eu; \
    if [ -d "${SGLANG_DIR}/sgl-kernel" ]; then \
        KERNEL_DIR="${SGLANG_DIR}/sgl-kernel"; \
    elif [ -d "${SGLANG_DIR}/python/sglang/kernels/aot" ]; then \
        KERNEL_DIR="${SGLANG_DIR}/python/sglang/kernels/aot"; \
    else \
        echo "ERROR: neither sgl-kernel/ nor python/sglang/kernels/aot exists" >&2; \
        exit 1; \
    fi; \
    echo "building kernels from ${KERNEL_DIR}"; \
    cd "${KERNEL_DIR}"; \
    if [ -f pyproject_rocm.toml ]; then \
        rm -f pyproject.toml; \
        mv pyproject_rocm.toml pyproject.toml; \
    fi; \
    AMDGPU_TARGET="${GPU_ARCHS}" python3 setup_rocm.py install

RUN set -eu; \
    cd "${SGLANG_DIR}"; \
    if [ ! -f python/pyproject_other.toml ]; then \
        echo "ERROR: python/pyproject_other.toml is missing" >&2; \
        exit 1; \
    fi; \
    python3 -m pip list --format=freeze | grep -E '^(torch|triton)' > /tmp/constraints.txt; \
    echo "pinning through the install:"; cat /tmp/constraints.txt; \
    if [ "${SGLANG_EXTRAS}" = "auto" ]; then \
        TORCH_MM="$(python3 -c 'import torch; print(".".join(torch.__version__.split(".")[:2]))')"; \
        case "${TORCH_MM}" in \
            2.9|2.10) EXTRAS=all_hip ;; \
            *)        EXTRAS=all_hip_rocm724 ;; \
        esac; \
        echo "base torch ${TORCH_MM} -> installing sglang[${EXTRAS}]"; \
    else \
        EXTRAS="${SGLANG_EXTRAS}"; \
    fi; \
    rm -f python/pyproject.toml; \
    mv python/pyproject_other.toml python/pyproject.toml; \
    python3 -m pip install --no-cache-dir -c /tmp/constraints.txt -e "python[${EXTRAS}]"; \
    if [ -n "${SGL_EVAL_VERSION}" ]; then \
        python3 -m pip install --no-cache-dir -c /tmp/constraints.txt \
            "sgl-eval==${SGL_EVAL_VERSION}"; \
    fi; \
    if python3 -m pip list --format=freeze | grep -Eq '^nvidia-.*-cu[0-9]+'; then \
        echo "ERROR: CUDA runtime packages were pulled into the ROCm image" >&2; \
        python3 -m pip list --format=freeze | grep -E '^nvidia-' >&2; \
        exit 1; \
    fi; \
    python3 -c "import torch; assert torch.version.hip, torch.__version__; print('torch', torch.__version__, 'HIP', torch.version.hip)"

# 90596736 is before aiter#5988 and does not contain aiter#7441.
# The patch adds only that pull request's two GEMM tables, then AITER
# is reinstalled from the patched tree.
COPY aiter_7441.patch /tmp/aiter_7441.patch
RUN set -eu; \
    python3 -m pip uninstall -y amd-aiter; \
    rm -rf "${AITER_DIR}"; \
    git clone --branch main --recursive "${AITER_REPO}" "${AITER_DIR}"; \
    cd "${AITER_DIR}"; \
    git checkout --detach "${AITER_REV}"; \
    got="$(git rev-parse HEAD)"; \
    case "${got}" in \
        "${AITER_REV}"*) ;; \
        *) echo "ERROR: asked for AITER ${AITER_REV}, got ${got}" >&2; exit 1 ;; \
    esac; \
    git submodule sync --recursive; \
    git submodule update --init --recursive; \
    test ! -f aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv; \
    git apply --check /tmp/aiter_7441.patch; \
    git apply /tmp/aiter_7441.patch; \
    test -f aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv; \
    test -f aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_untuned_gemm.csv; \
    rm -rf aiter/jit/build aiter/jit/__pycache__; \
    rm -f aiter/jit/*.so; \
    python3 -m pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt; \
    AITER_REBUILD=1 python3 setup.py develop; \
    python3 -c "from aiter.ops.flydsl import QuickAllReduceInt4; print('QuickAllReduceInt4', QuickAllReduceInt4)"; \
    echo "aiter ${got} plus aiter#7441"

RUN set -eu; \
    cd "${SGLANG_DIR}"; \
    got="$(git rev-parse HEAD)"; \
    case "${got}" in \
        "${SGLANG_REV}"*) ;; \
        *) echo "ERROR: collector image expected ${SGLANG_REV}, got ${got}" >&2; exit 1 ;; \
    esac; \
    test -f python/sglang/srt/layers/attention/linear/kernels/gdn_flydsl.py; \
    grep -q "linear-attn-prefill-backend flydsl" \
        python/sglang/srt/layers/attention/linear/gdn_backend.py; \
    echo "collector_19_11 ${got}"

LABEL org.opencontainers.image.base.name="${BASE_IMAGE}" \
      ai.amd.sglang.repo="${SGLANG_REPO}" \
      ai.amd.sglang.branch="${SGLANG_BRANCH}" \
      ai.amd.sglang.commit="${SGLANG_REV}" \
      ai.amd.aiter.repo="${AITER_REPO}" \
      ai.amd.aiter.commit="${AITER_REV}" \
      ai.amd.sglang.arm="collector-19-11"

WORKDIR /sgl-workspace/sglang
```

`docker/collector_19_11/aiter_7441.patch`:

```diff
commit ec91f61f0c550811cf121f0b0dc667c09524b161
Author:     Mikko Tukiainen <mikko.tukiainen@amd.com>
AuthorDate: Thu Nov 5 05:41:31 2026 +0200
Commit:     GitHub <noreply@github.com>
CommitDate: Thu Nov 5 11:41:31 2026 +0800

    Add gfx950 a4w4 GEMM configs for Qwen3.8-2.4T's GDN in_proj_qkvz at TP8 (#7441)
    
    N=4608, K=8192, 26 values of M from 1 to 16384, tuned with
    gemm_a4w4_blockscale_tune.py. An ASM kernel wins at every M, with error
    ratio 0. vLLM's AITER MXFP4 linear takes the ASM FP4 GEMM only for shapes
    tuned here, so this projection moves off the Triton GEMM.
    
    Signed-off-by: Mikko Tukiainen <Mikko.Tukiainen@amd.com>
    Co-authored-by: Cursor <cursoragent@cursor.com>

diff --git a/aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv b/aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv
new file mode 100644
index 00000000..2bc4cd88
--- /dev/null
+++ b/aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv
@@ -0,0 +1,27 @@
+gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
+gfx950,256,1,4608,8192,21,0,12.8814,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,5.86,1466.28,0.0
+gfx950,256,2,4608,8192,21,0,13.0145,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,11.6,1452.3,0.0
+gfx950,256,4,4608,8192,21,0,13.7312,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,21.99,1378.44,0.0
+gfx950,256,8,4608,8192,21,0,14.0,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,43.14,1355.78,0.0
+gfx950,256,16,4608,8192,21,0,13.7937,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,87.57,1383.77,0.0
+gfx950,256,32,4608,8192,21,0,14.7886,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,163.36,1305.08,0.0
+gfx950,256,48,4608,8192,21,0,14.4768,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,250.32,1347.9,0.0
+gfx950,256,64,4608,8192,21,0,14.4303,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,334.84,1367.01,0.0
+gfx950,256,80,4608,8192,21,0,13.8809,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,435.12,1436.46,0.0
+gfx950,256,96,4608,8192,21,0,14.9973,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,483.27,1343.73,0.0
+gfx950,256,112,4608,8192,21,0,14.506,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,582.91,1403.92,0.0
+gfx950,256,128,4608,8192,21,0,14.9112,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,648.08,1380.06,0.0
+gfx950,256,144,4608,8192,21,0,14.7405,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,737.54,1410.49,0.0
+gfx950,256,160,4608,8192,21,0,14.3645,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,840.93,1462.24,0.0
+gfx950,256,176,4608,8192,21,0,14.82,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,896.6,1431.67,0.0
+gfx950,256,192,4608,8192,21,0,14.3398,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,1010.86,1494.46,0.0
+gfx950,256,208,4608,8192,21,0,14.9174,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,1052.7,1450.87,0.0
+gfx950,256,224,4608,8192,21,0,15.1367,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E,1117.25,1443.92,0.0
+gfx950,256,240,4608,8192,29,0,15.885,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E,1140.66,1389.31,0.0
+gfx950,256,256,4608,8192,29,0,16.3101,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E,1184.99,1366.16,0.0
+gfx950,256,512,4608,8192,37,0,20.5976,_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128E,1876.66,1247.24,0.0
+gfx950,256,1024,4608,8192,46,0,28.1637,_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128E,2745.0,1154.18,0.0
+gfx950,256,2048,4608,8192,47,0,45.7475,_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256E,3379.83,1008.52,0.0
+gfx950,256,4096,4608,8192,47,0,85.1955,_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256E,3629.74,861.55,0.0
+gfx950,256,8192,4608,8192,52,0,154.1589,_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256E,4011.93,829.83,0.0
+gfx950,256,16384,4608,8192,54,0,267.4021,_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E,4625.81,886.22,0.0
diff --git a/aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_untuned_gemm.csv b/aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_untuned_gemm.csv
new file mode 100644
index 00000000..9285eab0
--- /dev/null
+++ b/aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_untuned_gemm.csv
@@ -0,0 +1,27 @@
+M,N,K
+1,4608,8192
+2,4608,8192
+4,4608,8192
+8,4608,8192
+16,4608,8192
+32,4608,8192
+48,4608,8192
+64,4608,8192
+80,4608,8192
+96,4608,8192
+112,4608,8192
+128,4608,8192
+144,4608,8192
+160,4608,8192
+176,4608,8192
+192,4608,8192
+208,4608,8192
+224,4608,8192
+240,4608,8192
+256,4608,8192
+512,4608,8192
+1024,4608,8192
+2048,4608,8192
+4096,4608,8192
+8192,4608,8192
+16384,4608,8192
```

# SGLang: Fast inference for LLMs and multimodal models

<p align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang" width="400">
</p>

<p align="center">
  <a href="https://pypi.org/project/sglang/"><img src="https://img.shields.io/pypi/v/sglang?style=flat&amp;label=PyPI&amp;labelColor=555555&amp;color=orange" alt="PyPI version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat&amp;labelColor=555555" alt="License: Apache 2.0"></a>
  <a href="https://pypistats.org/packages/sglang"><img src="https://img.shields.io/pypi/dm/sglang?style=flat&amp;label=Downloads&amp;labelColor=555555&amp;color=blue" alt="PyPI downloads per month"></a>
</p>

<p align="center">
  <a href="https://docs.sglang.io/">Docs</a> |
  <a href="https://cookbook.sglang.io/">Cookbook</a> |
  <a href="https://www.sglang.io/">Website</a> |
  <a href="https://lmsys.org/blog/">Blog</a> |
  <a href="https://slack.sglang.io/">Slack</a>
</p>

SGLang is an open-source inference framework for large language, vision-language, and diffusion models, optimized for agentic workloads, large-scale serving, and RL rollouts. [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion) is its built-in engine for image, video, audio, and action generation, included in this repository and the `sglang` Python package.

👋 Get started below, or meet the community at [SGLang Events](https://www.sglang.io/events), including meetups, developer meetings, workshops, and office hours.

> **[SGLang Summit 2026: Meet the builders of open AI](https://www.sglang.io/summit)**<br>
> **Our flagship annual gathering · November 12–13 · San Francisco**
>
> Go deep on inference, training, and AI hardware through technical talks, live demos, and hands-on workshops. Bring your deployment challenges and meet the engineers and researchers building the stack.

## Get Started

Pull the Docker image, which includes SGLang and its dependencies:

```bash
docker pull lmsysorg/sglang:latest
```

Alternatively, install SGLang in an activated Python environment with uv:

```bash
uv pip install --prerelease=allow sglang
```

Next, launch your model:

- [Quickstart](https://docs.sglang.io/docs/get-started/quickstart): Run your first model and send a request.
- [Cookbook](https://cookbook.sglang.io/): Choose your model and hardware to get a ready-to-run launch command.

## Supported Hardware

SGLang supports a wide range of GPUs, TPUs, NPUs, CPUs, and Apple Silicon platforms.

| Platform | Representative hardware |
| --- | --- |
| [NVIDIA](https://docs.sglang.io/docs/hardware-platforms/nvidia-gpus) | B200/B300/GB200/GB300; H100/H200/H800/H20; A100; RTX 30/40/50 series, RTX 6000 Ada / PRO 6000; DGX Spark, Jetson Orin |
| [AMD](https://docs.sglang.io/docs/hardware-platforms/amd_gpu) | Instinct MI300X, MI325X, MI350X, MI355X |
| [Google TPU](https://docs.sglang.io/docs/hardware-platforms/tpu) | v6e, v7; [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) / [SGL-JAX](https://github.com/sgl-project/sglang-jax) |
| Intel ([GPU](https://docs.sglang.io/docs/hardware-platforms/xpu) / [CPU](https://docs.sglang.io/docs/hardware-platforms/cpu_server)) | Arc / Arc Pro B-Series GPUs, Xeon CPUs |
| [Apple Silicon](https://docs.sglang.io/docs/hardware-platforms/apple_metal) | Macs via Metal / MLX |
| [Huawei Ascend](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation) | A2, A3, 950PR/DT NPUs |
| [Moore Threads](https://docs.sglang.io/docs/hardware-platforms/mthreads_gpu) | MTT S5000 GPUs |

Integrations in progress: AWS Trainium, [Alibaba T-Head PPU](https://github.com/sgl-project/sglang/issues/37519), [Cambricon MLU](https://github.com/sgl-project/sglang/pull/26898), Qualcomm QAIC, MetaX, Hygon HCU/DCU, Iluvatar CoreX, and more.

SGLang-TorchTPU brings the SGLang serving engine to Google TPUs through PyTorch, torch_tpu, and Pallas kernels. It supports LLM and vision-language serving on TPU7x (Ironwood), including tensor and expert parallelism across hosts. SGL-JAX provides the JAX-based TPU serving path.

See the [Cookbook](https://cookbook.sglang.io/) and platform guides for model compatibility and setup.

## SGL Ecosystem

| Area | Projects | Purpose |
| --- | --- | --- |
| Diffusion | [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion/installation) | Built into SGLang for image, video, audio, and action generation. |
| Audio | [SGLang Omni](https://github.com/sgl-project/sglang-omni) | Audio model serving for text-to-speech (TTS) and automatic speech recognition (ASR). |
| Speculative Decoding | [SpecForge](https://github.com/sgl-project/SpecForge) | Train draft models for speculative decoding and deploy them with SGLang. |
| RL and Post-Training | [Miles](https://github.com/radixark/miles), [slime](https://github.com/THUDM/slime), [AReaL](https://github.com/inclusionAI/AReaL), [Tunix](https://github.com/google/tunix), [verl](https://github.com/volcengine/verl) | Training frameworks that integrate SGLang for rollout generation. |
| Deployment and Orchestration | [RBG](https://github.com/sgl-project/rbg), [llm-d](https://llm-d.ai/docs/dev/operations/disaggregation/sglang), [Ray Serve](https://docs.ray.io/en/latest/serve/llm/user-guides/sglang.html), [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/backends/sg-lang/reference-guide) | Deploy and scale SGLang inference services with routing, load balancing, and cluster orchestration. |
| Education | [Mini-SGLang](https://github.com/sgl-project/mini-sglang), [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang), [DeepLearning.AI course](https://www.deeplearning.ai/short-courses/efficient-inference-with-sglang-text-and-image-generation/) | Learn inference engine design and efficient text and image generation through code and hands-on courses. |

## Development and Contributing

Contributions are welcome, from bug fixes and documentation to model support and performance improvements.

### Development setup

Start from the `lmsysorg/sglang:dev` Docker image, which provides development tools and most dependencies. Clone or mount your SGLang checkout inside the container, then install it in editable mode from the repository root so tests use your local Python changes:

```bash
pip install -e "python"
```

In an activated virtual environment, you can use `uv pip install --prerelease=allow -e "python"` instead. See the [development guide](https://docs.sglang.io/docs/developer_guide/development_guide_using_docker) for container setup and testing.

### Contribute

1. Fork the repository and create a branch for your changes. For larger changes, discuss your proposal in a [GitHub issue](https://github.com/sgl-project/sglang/issues) or on [Slack](https://slack.sglang.io/).
2. Make your changes, run the relevant tests, and add regression coverage for fixes or new behavior. Run `pre-commit run --all-files` before submitting.
3. Open a pull request describing the change and how you tested it. Include benchmarks or accuracy evaluations when relevant.

See the [contributor guide](https://docs.sglang.io/docs/developer_guide/contribution_guide) for formatting, testing, and pull request instructions. Documentation contributors can start with the [docs guide](docs/README.md).

## Community and Sponsorship

SGLang is hosted by [LMSYS](https://lmsys.org/about/), a non-profit open-source organization.

- **Community discussions:** Join [Slack](https://slack.sglang.io/) for technical questions and development discussions.
- **Events:** Find meetups, workshops, and office hours on [SGLang Events](https://www.sglang.io/events).
- **Updates:** Follow [X](https://x.com/lmsysorg) and [LinkedIn](https://www.linkedin.com/company/sgl-project/) for project updates, and the [LMSYS Blog](https://lmsys.org/blog/) for release announcements and technical articles.
- **Project resources:** Explore the [documentation](https://docs.sglang.io/), [Cookbook](https://cookbook.sglang.io/), [roadmap](https://roadmap.sglang.io/), [release notes](https://github.com/sgl-project/sglang/releases), [issue tracker](https://github.com/sgl-project/sglang/issues), and [contributor guide](https://docs.sglang.io/docs/developer_guide/contribution_guide).
- **Contact Us:** For enterprise adoption and deployment, technical consulting, sponsorship, or partnership inquiries, please contact [sglang@lmsys.org](mailto:sglang@lmsys.org).
- **Contributor sponsorship:** Long-term active SGLang contributors are eligible for coding agent sponsorship, including Cursor, Claude Code, or OpenAI Codex. To apply, email [sglang@lmsys.org](mailto:sglang@lmsys.org) with links to your key commits or pull requests.

## Trusted by Industry and Research

SGLang serves production workloads across AI labs, cloud platforms, enterprises, and universities.

<img src="assets/adoption.png" alt="Organizations adopting SGLang" width="800">

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
