# collector_261128_qwen_3.8_mono_h32: the 20261127 SGLang nightly with sgl-project/sglang#46059 (the Hadamard-32
# rotation of the online MXFP4 GDN projections, on by default) and the sgl-amd-mono plugin at PLUGIN_SHA of
# omirosh/sgl-amd-mono: q38v3/15-h32: the stack q38v3/13-stack with the split and wide launches on the rotated
# projections (mxfp4_h32) in place of the rotation guard.
# The base AITER has the rotation of per_1x32_f4_quant_hip (ROCm/aiter#6734). No AITER change.
# Build: docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_h32 collector/
ARG BASE_IMAGE=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SGLANG_BASE=102a17d4df19be4941d9e5ebb73418b00a22e347
ARG SGLANG_REPO=https://github.com/amd-oshkarav/sglang.git
ARG SGLANG_REV=8d612e062afa14ad798652c448fa6fad615a42b5
ARG AITER_REV=581579ab51ca26ddd4a08a6aa6aa983a78aaf276
ARG PLUGIN_REPO=https://github.com/omirosh/sgl-amd-mono.git
ARG PLUGIN_SHA=5bea396cc304312d44d80914c78724c2abf6b680
LABEL ai.amd.base_image="${BASE_IMAGE}" ai.amd.sglang="${SGLANG_REPO}@${SGLANG_REV}" ai.amd.aiter="${AITER_REV}" \
      ai.amd.sgl_amd_mono="${PLUGIN_REPO}@${PLUGIN_SHA}" ai.amd.sglang_prs="sgl-project/sglang#46059@122da99f69"

RUN set -eux; \
    git config --global --add safe.directory '*'; \
    test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_BASE}"; \
    test "$(git -C /sgl-workspace/aiter rev-parse HEAD)" = "${AITER_REV}"; \
    cd /sgl-workspace/sglang; \
    git fetch "${SGLANG_REPO}" "${SGLANG_REV}"; \
    git checkout -f "${SGLANG_REV}"; \
    test "$(git rev-parse HEAD)" = "${SGLANG_REV}"; \
    python3 -c "import sglang, os; assert os.path.realpath(sglang.__file__).startswith('/sgl-workspace/sglang/python/'), sglang.__file__"; \
    python3 -c "from sglang.srt.environ import envs; assert envs.SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD.get()"; \
    python3 -c "import inspect; from aiter import per_1x32_f4_quant_hip as q; assert 'hadamard_rotate' in inspect.signature(q).parameters"; \
    cd /sgl-workspace/sgl-amd-mono; \
    git fetch "${PLUGIN_REPO}" "${PLUGIN_SHA}"; \
    git checkout -f "${PLUGIN_SHA}"; \
    test "$(git rev-parse HEAD)" = "${PLUGIN_SHA}"; \
    python3 -m pip install --no-cache-dir --no-deps -e .; \
    python3 -c "import sgl_amd_mono; assert 'Qwen3_5MoeForCausalLM' in {k[0] if isinstance(k, tuple) else k for k in sgl_amd_mono.SPECS}"; \
    python3 -c "from sgl_amd_mono.qwen38_a95b_mxfp4 import spec; assert hasattr(spec, '_wide_format')"; \
    mkdir -p /etc/sglang-build; \
    { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "sglang=${SGLANG_REPO}@${SGLANG_REV} (${SGLANG_BASE} + sgl-project/sglang#46059)"; \
      echo "aiter=${AITER_REV}"; \
      echo "sgl_amd_mono=${PLUGIN_SHA}"; \
    } >> /etc/sglang-build/manifest.txt; \
    cat /etc/sglang-build/manifest.txt; \
    python3 -m pip list --format=freeze > /sgl-workspace/pip_freeze.txt
