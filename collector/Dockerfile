# collector_261127_qwen_3.8_best_aiter8829: the image of collector_24_11 (the best run of
# week 47 on a newer base) without sglang#39069, plus aiter#8829.
# Build: docker build --network=host -t amdsiloai/sglang-private:<branch> collector/
#
# Three layers on lmsysorg/sglang:v0.5.23-rocm10-mi35x (pinned by digest), as in collector_24_11:
#
# 1. SGLang. amd-oshkarav/sglang collector_261127_qwen_3.8_best_aiter8829 @ SGLANG_REV: sgl-project
#    main 88dc7501f5 plus sglang#40771, FlyDSL INT4 QuickReduce (8cb39392), sglang#47219,
#    sglang#47433, sglang#45529 and sglang#39595. These are the commits of collector_24_11 without
#    sglang#39069 (the draft path of bundled MTP; this image is for the non-MTP config).
#
# 2. AITER. ROCm/aiter main @ AITER_REV (as collector_24_11) plus aiter_pr8829.patch
#    (aiter#8829: the stage2 FP8 row table also for AITER_MXFP4_INTERMEDIATE=1). aiter_setup.sh:
#    - Applies the patch with git apply --check, then git apply. No other edit.
#    - AITER_USE_SYSTEM_TRITON=1 keeps the Triton of the base image. The build fails if the
#      AITER step changes the Triton packages.
#    - Checks the AITER functions that the FlyDSL GDN prefill backend imports, and imports
#      QuickAllReduceInt4 and GateMode.
#
# 3. sgl-amd-mono. main @ 4cba5b25 plus sgl-amd-mono#154 (head 05b9dd9f), as collector_24_11.

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x@sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG SGLANG_REPO=https://github.com/amd-oshkarav/sglang.git
ARG SGLANG_BRANCH=collector_261127_qwen_3.8_best_aiter8829
ARG SGLANG_DIR=/sgl-workspace/sglang
ARG SGLANG_REV=271511b7fae0e44279e8037136a9f0070b97a877
ARG SGLANG_EXTRAS=auto
ARG SGL_EVAL_VERSION=0.1.2
ARG AITER_DIR=/sgl-workspace/aiter
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REV=4120f98392c0f4437b572b5f131a62444c125435
# aiter#8829 at this head, applied as aiter_pr8829.patch (its diff from the merge base with main).
ARG AITER_PR8829_HEAD=688913c38d18e672a2f90c1a588ca73f76e03b90
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
        echo "ERROR: the base image has triton ${got:-none}, this image needs ${EXPECT_TRITON}." >&2; \
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

# AITER at AITER_REV plus aiter#8829. The step is in aiter_setup.sh and is copied in: ITT's classic builder does not
# feed a RUN heredoc to the command.
COPY aiter_setup.sh aiter_pr8829.patch /tmp/
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
    if grep -q "bundles its MTP draft" python/sglang/srt/arg_groups/speculative_hook.py; then \
        echo "ERROR: sglang#39069 is in the tree" >&2; exit 1; \
    fi; \
    echo "collector_261127_qwen_3.8_best_aiter8829 ${got}"

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
      ai.amd.aiter.patch="aiter#8829 ${AITER_PR8829_HEAD}" \
      ai.amd.sglang.arm="collector-261127-best-aiter8829"

RUN python3 -m pip list --format=freeze > /sgl-workspace/pip_freeze.txt

WORKDIR /sgl-workspace/sglang
