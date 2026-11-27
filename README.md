[example parent sweep](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/a98a78f50fa54ae7b77d81a98035e128?o=4032599967441609)

## 1. SGLang: code injected on top of main

Branch: https://github.com/amd-oshkarav/sglang/tree/collector_24_11, pinned to
`c4a2edc9244bb2ce80c24502cb31b6a0256d33fc`. It starts from sgl-project/sglang main `88dc7501f58f026a18357b8ad4ef1f280e1fc2b4` 

| Commit | Pull request | Feature | Files changed |
| --- | --- | --- | --- |
| `a756297335` | [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP | `arg_groups/speculative_hook.py` |
| `665ddf2c74` | [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce | `distributed/parallel_state.py` |
| `3e49b12043` | [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | Use FlyDSL INT4 QuickReduce for contiguous bf16 | `device_communicators/quick_all_reduce.py`, 2 unit tests |
| `5ef3f38228` | [#47219](https://github.com/sgl-project/sglang/pull/47219) | Fuse add-RMSNorm into FlyDSL INT4 QuickReduce | `quick_all_reduce.py`, `parallel_state.py`, `layer_boundary/residual/add_norm.py`, unit test |
| `7e8d8d53ae` | [#47433](https://github.com/sgl-project/sglang/pull/47433) | Use one ReplaySSM pipeline stage on gfx950 | `gdn_replayssm_spec_decode.py` |
| `4f1547fd70` | [#45529](https://github.com/sgl-project/sglang/pull/45529) | Fuse self-attn prepare on gfx950 (norm, RoPE, gate) | `fused_qk_rmsnorm_rope_gate.py`, `models/qwen3_5.py` |
| `4af7130f0d` | [#39595](https://github.com/sgl-project/sglang/pull/39595) | Add the FlyDSL GDN prefill backend (`--linear-attn-prefill-backend flydsl`) | `arg_groups/choices.py`, `linear/gdn_backend.py`, new `linear/kernels/gdn_flydsl.py`, `linear/utils.py`, `forward_batch_info.py`, `eager_runner.py` |



## 2. AITER

`ROCm/aiter` main @ `4120f98392c0f4437b572b5f131a62444c125435` (2026-11-26). 


## 3. sgl-amd-mono

- Repository: https://github.com/sgl-project/sgl-amd-mono
- Base: `main` @ `4cba5b253e7c2bbe8329d4f423538a6474083514`.
- PR: [#154](https://github.com/sgl-project/sgl-amd-mono/pull/154), "[Perf] [MI355X] qwen38_a95b_mxfp4: Run the tail of 16-row steps as one layer_block launch" (by omirosh).
  Five commits from the merge base with the base commit up to the head `05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68`, cherry-picked in order.
  The list is in `/sgl-workspace/sgl_amd_mono_picks.txt` inside the image.
- PR 147 is not cherry-picked. Its change is in main since `36c385e`.
- The package is installed with `pip install --no-deps -e .` after SGLang, and the build checks that the `sglang.srt.plugins` entry point `sgl_amd_mono` is registered.
  The server enables it with `--enable-amd-monokernel`.
- Build args: `SGL_AMD_MONO_REV`, `SGL_AMD_MONO_PRS=154`, `SGL_AMD_MONO_PR_REVS='154=05b9dd9f...'`.


## 4. Dockerfile and script (verbatim)

### 4.1 `Dockerfile`
```bash
# Clean AITER image for the collector_24_11 branch: ROCm/aiter main head of
# 2026-11-26, 4120f9839 ("[FlyDSL] [gfx1250] grouped MoE GEMM1 opts into mega
# (#7921)"), as it is. No patch, no table copy, no edit of AITER.
#
# Triton: the base image's Triton 3.8.0 is kept (AITER_USE_SYSTEM_TRITON=1,
# EXPECT_TRITON below). AITER's own Triton is 3.9.0 at this head (aiter#8266).
#
# Three layers on lmsysorg/sglang:v0.5.23-rocm10-mi35x, as in the week 47 image:
#
# 1. SGLang. amd-oshkarav/sglang collector_24_11 @ c4a2edc924 (sgl-project main
#    88dc7501f5 plus #39069, #40771, FlyDSL INT4 QuickReduce, #47219, #47433,
#    #45529 and #39595). Cloned and reinstalled on every step. Nothing is patched.
#
# 2. AITER. ROCm/aiter main @ AITER_REV, as it is (aiter_setup.sh):
#    - No patch, no table copy, no table edit. Whatever the commit has is used.
#    - AITER_USE_SYSTEM_TRITON=1 keeps the Triton of the base image. The build
#      fails if the AITER step changes the Triton packages.
#    - The build checks the AITER functions that the FlyDSL GDN prefill backend
#      imports, and imports QuickAllReduceInt4 and GateMode.
#
# 3. sgl-amd-mono. main @ 4cba5b25 plus PR 154 (head 05b9dd9f), same as the
#    week 47 image.
#
#   docker build --network=host --progress=plain \
#       -f docker/collector_24_11_aiter_clean/Dockerfile \
#       --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x \
#       --build-arg AITER_REV=4120f98392c0f4437b572b5f131a62444c125435 \
#       -t <tag> docker/collector_24_11_aiter_clean
#
# ITT needs dockerfile.path to be this file:
#   /app/repos/inference-testing/docker/collector_24_11_aiter_clean/Dockerfile

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG SGLANG_REPO=https://github.com/amd-oshkarav/sglang.git
ARG SGLANG_BRANCH=collector_24_11
ARG SGLANG_DIR=/sgl-workspace/sglang
ARG SGLANG_REV=c4a2edc9244bb2ce80c24502cb31b6a0256d33fc
ARG SGLANG_EXTRAS=auto
ARG SGL_EVAL_VERSION=0.1.2
ARG AITER_DIR=/sgl-workspace/aiter
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REV=4120f98392c0f4437b572b5f131a62444c125435
# 1 keeps the base image's Triton during the AITER build.
ARG AITER_USE_SYSTEM_TRITON=1
# The Triton of the week 47 image. The build stops when the base differs.
ARG EXPECT_TRITON=3.8.0+git4cff872c.rocm10.0.0
ARG SGL_AMD_MONO_DIR=/sgl-workspace/sgl-amd-mono
ARG SGL_AMD_MONO_REPO=https://github.com/sgl-project/sgl-amd-mono.git
ARG SGL_AMD_MONO_REV=4cba5b253e7c2bbe8329d4f423538a6474083514
# PR numbers, space separated, cherry-picked in this order. Empty: none.
ARG SGL_AMD_MONO_PRS=154
# "<pr>=<head sha>" pairs, space separated. A PR without an entry is taken at
# its head when the image is built.
ARG SGL_AMD_MONO_PR_REVS="154=05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68"

ENV GPU_ARCHS=gfx950
ENV PYTORCH_ROCM_ARCH=gfx950
ENV AMDGPU_TARGETS=gfx950
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_BUILD_JOBS=4

WORKDIR /sgl-workspace

# Triton of the base image. Cheap, and first, so a base with another Triton
# stops the build before the long steps.
RUN set -eu; \
    got="$(python3 -m pip show triton 2>/dev/null | sed -n 's/^Version: //p')"; \
    echo "base image triton: ${got:-none}"; \
    python3 -m pip show triton triton-kernels || true; \
    if [ -n "${EXPECT_TRITON}" ] && [ "${got}" != "${EXPECT_TRITON}" ]; then \
        echo "ERROR: the base image has triton ${got:-none}, the bisection images need ${EXPECT_TRITON}." >&2; \
        echo "Use a base with that Triton, or set --build-arg EXPECT_TRITON= to skip this check." >&2; \
        exit 1; \
    fi

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

# AITER at AITER_REV, as it is. The step is in aiter_setup.sh and is copied in: ITT's classic builder does not
# feed a RUN heredoc to the command.
COPY aiter_setup.sh /tmp/aiter_setup.sh
RUN bash /tmp/aiter_setup.sh

# The feature tree must be the pinned commit and must carry the collector
# features that the config uses.
RUN set -eu; \
    cd "${SGLANG_DIR}"; \
    got="$(git rev-parse HEAD)"; \
    case "${got}" in \
        "${SGLANG_REV}"*) ;; \
        *) echo "ERROR: feature image expected ${SGLANG_REV}, got ${got}" >&2; exit 1 ;; \
    esac; \
    test -f python/sglang/srt/layers/attention/linear/kernels/gdn_flydsl.py; \
    grep -q '"flydsl"' python/sglang/srt/arg_groups/choices.py; \
    grep -q "linear-attn-prefill-backend flydsl" \
        python/sglang/srt/layers/attention/linear/gdn_backend.py; \
    echo "collector_24_11 ${got}"

# --enable-amd-monokernel loads the sgl_amd_mono entry point. Installed
# after sglang so an extra cannot put another copy back. Each PR is the
# commits from its merge base with SGL_AMD_MONO_REV up to the pinned head.
RUN set -eu; \
    git config --global --add safe.directory '*'; \
    python3 -m pip uninstall -y sgl-amd-mono 2>/dev/null || true; \
    rm -rf "${SGL_AMD_MONO_DIR}"; \
    git clone --filter=blob:none --branch main \
        "${SGL_AMD_MONO_REPO}" "${SGL_AMD_MONO_DIR}"; \
    cd "${SGL_AMD_MONO_DIR}"; \
    git checkout --detach "${SGL_AMD_MONO_REV}"; \
    got="$(git rev-parse HEAD)"; \
    case "${got}" in \
        "${SGL_AMD_MONO_REV}"*) ;; \
        *) echo "ERROR: asked for sgl-amd-mono ${SGL_AMD_MONO_REV}, got ${got}" >&2; exit 1 ;; \
    esac; \
    python3 -c "import inspect; from sglang.kernels.plugin_kernels.amd.spec import register; assert 'model_type' in inspect.signature(register).parameters"; \
    expected=0; \
    for pr in ${SGL_AMD_MONO_PRS}; do \
        git fetch origin "pull/${pr}/head"; \
        head="$(git rev-parse FETCH_HEAD)"; \
        pin="$(printf '%s\n' ${SGL_AMD_MONO_PR_REVS} | sed -n "s/^${pr}=//p")"; \
        if [ -n "${pin}" ]; then \
            git cat-file -e "${pin}^{commit}" || { echo "ERROR: PR ${pr} has no commit ${pin} (head is ${head})" >&2; exit 1; }; \
            tip="${pin}"; \
        else \
            tip="${head}"; \
        fi; \
        base="$(git merge-base "${SGL_AMD_MONO_REV}" "${tip}")"; \
        commits="$(git rev-list --reverse --no-merges "${base}..${tip}")"; \
        if [ -z "${commits}" ]; then \
            echo "ERROR: PR ${pr} has no commits above ${base}" >&2; \
            exit 1; \
        fi; \
        echo "PR ${pr}: head ${head}, picking up to ${tip} from merge base ${base}:"; \
        git log --no-walk=unsorted --format='  %h %s' ${commits}; \
        git -c user.email=docker-build@local -c user.name='docker build' \
            cherry-pick ${commits}; \
        expected=$((expected + $(printf '%s\n' "${commits}" | wc -l))); \
    done; \
    git merge-base --is-ancestor "${SGL_AMD_MONO_REV}" HEAD; \
    n="$(git rev-list --count "${SGL_AMD_MONO_REV}..HEAD")"; \
    test "${n}" = "${expected}"; \
    spec=sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    test "$(grep -c state_layer_cache "${spec}")" -ge 1; \
    test "$(grep -c mamba2_layer_cache "${spec}")" = 0; \
    python3 -m pip install --no-cache-dir --no-deps -e .; \
    python3 -c "from importlib.metadata import entry_points; assert any(e.name == 'sgl_amd_mono' for e in entry_points(group='sglang.srt.plugins')), 'sgl_amd_mono plugin not registered'"; \
    git log --format='%h %s' "${SGL_AMD_MONO_REV}..HEAD" > /sgl-workspace/sgl_amd_mono_picks.txt; \
    echo "sgl-amd-mono main ${SGL_AMD_MONO_REV} plus PRs '${SGL_AMD_MONO_PRS}' (${n} commits), HEAD $(git rev-parse HEAD)"

LABEL org.opencontainers.image.base.name="${BASE_IMAGE}" \
      ai.amd.sglang.repo="${SGLANG_REPO}" \
      ai.amd.sglang.branch="${SGLANG_BRANCH}" \
      ai.amd.sglang.commit="${SGLANG_REV}" \
      ai.amd.aiter.repo="${AITER_REPO}" \
      ai.amd.aiter.commit="${AITER_REV}" \
      ai.amd.aiter.use_system_triton="${AITER_USE_SYSTEM_TRITON}" \
      ai.amd.sgl_amd_mono.repo="${SGL_AMD_MONO_REPO}" \
      ai.amd.sgl_amd_mono.commit="${SGL_AMD_MONO_REV}" \
      ai.amd.sgl_amd_mono.pr="${SGL_AMD_MONO_PRS}" \
      ai.amd.sgl_amd_mono.pr_revs="${SGL_AMD_MONO_PR_REVS}" \
      ai.amd.sglang.arm="collector-24-11-aiter-clean"

WORKDIR /sgl-workspace/sglang

```

### 4.2 `aiter_setup.sh`

```bash
#!/usr/bin/env bash
# AITER at ${AITER_REV}, as it is. No patch, no table copy, no edit.
# Copied in and run by the Dockerfile. The classic builder used by ITT does not
# feed a RUN heredoc to the command, so the step lives in this file.
# The build arguments are passed in the environment by the RUN line.
set -eu
python3 -m pip uninstall -y amd-aiter
rm -rf "${AITER_DIR}"
git clone --branch main --recursive "${AITER_REPO}" "${AITER_DIR}"
cd "${AITER_DIR}"
git checkout --detach "${AITER_REV}"
got="$(git rev-parse HEAD)"
case "${got}" in
    "${AITER_REV}"*) ;;
    *) echo "ERROR: asked for AITER ${AITER_REV}, got ${got}" >&2; exit 1 ;;
esac
git submodule sync --recursive
git submodule update --init --recursive
cfg="aiter/configs/model_configs"
fmoe="${cfg}/qwen3_8_2400b_fp4_tuned_fmoe.csv"
if [ ! -f "${fmoe}" ]; then
    echo "ERROR: ${fmoe} is missing at ${got}" >&2
    exit 1
fi
# Reported only. Nothing is edited.
q256n="$(grep -c 'fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1' "${fmoe}" || true)"
echo "Q256 MoE rows in ${fmoe}: ${q256n}"
if [ ! -e "${cfg}/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv" ]; then
    echo "NOTE: ${got} has no qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv (the build continues without it)"
fi
rm -rf aiter/jit/build aiter/jit/__pycache__
rm -f aiter/jit/*.so
triton_before="$(python3 -m pip list --format=freeze | grep -iE '^(triton|triton-kernels)==' | sort || true)"
echo "Triton packages before the AITER build: ${triton_before:-none}"
python3 -m pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt
AITER_USE_SYSTEM_TRITON="${AITER_USE_SYSTEM_TRITON}" AITER_REBUILD=1 python3 setup.py develop
triton_after="$(python3 -m pip list --format=freeze | grep -iE '^(triton|triton-kernels)==' | sort || true)"
echo "Triton packages after the AITER build: ${triton_after:-none}"
if [ "${AITER_USE_SYSTEM_TRITON}" = "1" ] && [ "${triton_before}" != "${triton_after}" ]; then
    echo "ERROR: the AITER build changed the Triton packages although AITER_USE_SYSTEM_TRITON=1" >&2
    exit 1
fi
# The AITER functions that SGLang collector_24_11 imports on the Qwen3.8 path (FlyDSL
# GDN prefill, causal-conv update, prefill metadata), checked in the source
# files. They are not imported here: the build host may have no GPU. A missing
# name stops the build now and not at server start.
for spec in \
    "aiter/ops/flydsl/linear_attention_prefill_kernels.py:def chunk_gated_delta_rule_fwd_h_flydsl_opt" \
    "aiter/ops/flydsl/linear_attention_prefill_kernels.py:def gdn_prepare_flydsl_supported" \
    "aiter/ops/flydsl/linear_attention_prefill_kernels.py:def gdn_prepare_fwd_flydsl" \
    "aiter/ops/flydsl/causal_conv1d_update_kernels.py:def causal_conv1d_update_sglang_flydsl" \
    "aiter/ops/prefill_batch_metadata.py:def build_gated_delta_rule_prefill_metadata"; do
    f="${spec%%:*}"
    pat="${spec#*:}"
    if ! grep -q "${pat}" "${f}"; then
        echo "ERROR: ${f} has no '${pat}' at ${got}" >&2
        exit 1
    fi
done
python3 - <<'PY'
# The all-reduce class and the MoE GateMode that SGLang imports must import.
# The optional names are only reported.
import importlib

required = [
    ("aiter.ops.flydsl", "QuickAllReduceInt4"),
    ("aiter.ops.flydsl.moe_common", "GateMode"),
]
optional = [
    ("aiter.ops.flydsl", "FlyQuickAllReduce"),
    ("aiter.ops.triton._triton_kernels.gated_delta_net.prefill.chunk_o", "chunk_fwd_o_opt_vk"),
    ("aiter.ops.triton._triton_kernels.gated_delta_net.utils", "l2norm_fwd"),
]
missing = []
for module, name in required:
    try:
        getattr(importlib.import_module(module), name)
        print(f"ok       {module}.{name}")
    except Exception as exc:
        print(f"MISSING  {module}.{name}: {exc!r}")
        missing.append(f"{module}.{name}")
for module, name in optional:
    try:
        getattr(importlib.import_module(module), name)
        print(f"present  {module}.{name}")
    except Exception as exc:
        print(f"absent   {module}.{name}: {exc!r}")
if missing:
    raise SystemExit("ERROR: missing AITER names: " + ", ".join(missing))
PY
python3 -m pip show triton triton-kernels > /sgl-workspace/pip_show_triton.txt 2>&1 || true
cat /sgl-workspace/pip_show_triton.txt
echo "aiter ${got} as is, no patch, Q256 rows ${q256n}"
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
