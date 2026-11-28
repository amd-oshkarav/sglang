# collector_261128_qwen_3.8_mono_draft: the 20261127 SGLang nightly with the sgl-amd-mono plugin at PLUGIN_SHA of
# omirosh/sgl-amd-mono, q38v3/21-draft: q38v3/19-all (the image collector_261128_qwen_3.8_mono_all) plus the MTP draft
# spec (qwen38_a95b_mxfp4/draft.py: K2 runs the tail of the draft layer in the draft steps of 1 to 8 rows), plus
# sgl-project/sglang#53137 at its head 8beb60f9be: the capture check of the runtime skips a draft runner. Without it,
# a draft spec stops the server startup (since sgl-project/sglang#46210).
# SGLANG_REV is the [Port] commit on this branch, on SGLANG_BASE of the base image. The build applies the diff
# SGLANG_BASE..SGLANG_REV to the editable SGLang tree of the base, which keeps the ROCm pyproject files of the nightly
# build, and checks that the 2 files are SGLANG_REV's.
# No AITER change. SGLANG_AMD_MONO_K1V=0 turns K1v off.
# Build: docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_draft collector/
ARG BASE_IMAGE=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SGLANG_BASE=102a17d4df19be4941d9e5ebb73418b00a22e347
ARG SGLANG_REPO=https://github.com/amd-oshkarav/sglang.git
ARG SGLANG_REV=032ce33b73bffe6b72f94f1147da69580d5a60f8
ARG PR53137_HEAD=8beb60f9be5f84e4b478059866bdff5ced76711a
ARG AITER_REV=581579ab51ca26ddd4a08a6aa6aa983a78aaf276
ARG PLUGIN_REPO=https://github.com/omirosh/sgl-amd-mono.git
ARG PLUGIN_SHA=3ddb968f06a9726c306102bddae01d58defed4c2
LABEL ai.amd.base_image="${BASE_IMAGE}" ai.amd.sglang="${SGLANG_REPO}@${SGLANG_REV}" ai.amd.aiter="${AITER_REV}" \
      ai.amd.sglang_pr53137="${PR53137_HEAD}" \
      ai.amd.sgl_amd_mono="${PLUGIN_REPO}@${PLUGIN_SHA}"

RUN set -eux; \
    git config --global --add safe.directory '*'; \
    test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_BASE}"; \
    git -C /sgl-workspace/sglang fetch "${SGLANG_REPO}" "${SGLANG_REV}"; \
    test "$(git -C /sgl-workspace/sglang rev-parse FETCH_HEAD)" = "${SGLANG_REV}"; \
    test "$(git -C /sgl-workspace/sglang rev-parse "${SGLANG_REV}~1")" = "${SGLANG_BASE}"; \
    files="python/sglang/srt/kernel_plugins/amd/binding.py test/registered/unit/kernel_plugins/amd/test_mono_framework.py"; \
    test "$(git -C /sgl-workspace/sglang diff --name-only "${SGLANG_BASE}" "${SGLANG_REV}" | sort | tr '\n' ' ')" = "${files} "; \
    git -C /sgl-workspace/sglang diff --quiet "${SGLANG_BASE}" -- ${files}; \
    git -C /sgl-workspace/sglang diff "${SGLANG_BASE}" "${SGLANG_REV}" | git -C /sgl-workspace/sglang apply; \
    git -C /sgl-workspace/sglang diff --quiet "${SGLANG_REV}" -- ${files}; \
    git -C /sgl-workspace/sglang status --short; \
    (cd / && python3 -c "import sglang.srt.kernel_plugins.amd.binding as m; assert m.__file__.startswith('/sgl-workspace/sglang/python/'), m.__file__"); \
    grep -q 'if getattr(runner.model_runner, "is_draft_worker", False):' /sgl-workspace/sglang/python/sglang/srt/kernel_plugins/amd/binding.py; \
    test "$(git -C /sgl-workspace/aiter rev-parse HEAD)" = "${AITER_REV}"; \
    cd /sgl-workspace/sgl-amd-mono; \
    git fetch "${PLUGIN_REPO}" "${PLUGIN_SHA}"; \
    git checkout -f "${PLUGIN_SHA}"; \
    test "$(git rev-parse HEAD)" = "${PLUGIN_SHA}"; \
    python3 -m pip install --no-cache-dir --no-deps -e .; \
    python3 -c "import sgl_amd_mono; assert 'Qwen3_5MoeForCausalLM' in {k[0] if isinstance(k, tuple) else k for k in sgl_amd_mono.SPECS}"; \
    python3 -c "import sgl_amd_mono; assert sgl_amd_mono.SPECS['Qwen3_5ForCausalLMMTP', None] == 'sgl_amd_mono.qwen38_a95b_mxfp4.draft:SPEC'"; \
    grep -q '^def _route_draft_steps(' sgl_amd_mono/qwen38_a95b_mxfp4/draft.py; \
    python3 -c "import inspect; from sglang.kernels.plugin_kernels.amd.spec import register; assert 'model_type' in inspect.signature(register).parameters"; \
    grep -q '^class Ring(NamedTuple):' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/gdn.py; \
    grep -q '^def _finish_gdn_replayssm_circular_fold_kernel(' /sgl-workspace/sglang/python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py; \
    grep -q '^UP_MAX = 3$' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/mono/blocks.py; \
    grep -q 'return "mxfp4_h32" if rotated' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    grep -q '^def shared_down(' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/wide/stages/moe.py; \
    grep -q '^LAYER_WIDTHS_F0 = (3, 6, 8)$' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    mkdir -p /etc/sglang-build; \
    { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "sglang_base=${SGLANG_BASE}"; \
      echo "sglang=${SGLANG_REPO}@${SGLANG_REV}"; \
      echo "sglang_pr53137=${PR53137_HEAD}"; \
      echo "aiter=${AITER_REV}"; \
      echo "sgl_amd_mono=${PLUGIN_SHA}"; \
    } >> /etc/sglang-build/manifest.txt; \
    cat /etc/sglang-build/manifest.txt; \
    python3 -m pip list --format=freeze > /sgl-workspace/pip_freeze.txt
