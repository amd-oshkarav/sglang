# collector_261128_qwen_3.8_mono_nofse: the 20261127 SGLang nightly with the sgl-amd-mono plugin at
# PLUGIN_SHA of omirosh/sgl-amd-mono: q38v3/18-nofse: the stack q38v3/13-stack (P6, sgl-amd-mono#156, #159, #174, #180,
# the rotation guard and D2) plus the monokernels without the fused shared expert: the split GDN launch at 16 rows,
# the wide launches at 32 rows (SGLang's bf16 shared expert in the wide MoE) and layer_block at 3, 6 and 8 rows.
# No SGLang change and no AITER change: the build checks the SGLang and AITER commits of the base.
# Build: docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_nofse collector/
ARG BASE_IMAGE=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SGLANG_REV=102a17d4df19be4941d9e5ebb73418b00a22e347
ARG AITER_REV=581579ab51ca26ddd4a08a6aa6aa983a78aaf276
ARG PLUGIN_REPO=https://github.com/omirosh/sgl-amd-mono.git
ARG PLUGIN_SHA=23fe99e1a1c61c9ebc3a93ed4c5dc3772c326b12
LABEL ai.amd.base_image="${BASE_IMAGE}" ai.amd.sglang="${SGLANG_REV}" ai.amd.aiter="${AITER_REV}" \
      ai.amd.sgl_amd_mono="${PLUGIN_REPO}@${PLUGIN_SHA}"

RUN set -eux; \
    git config --global --add safe.directory '*'; \
    test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_REV}"; \
    test "$(git -C /sgl-workspace/aiter rev-parse HEAD)" = "${AITER_REV}"; \
    cd /sgl-workspace/sgl-amd-mono; \
    git fetch "${PLUGIN_REPO}" "${PLUGIN_SHA}"; \
    git checkout -f "${PLUGIN_SHA}"; \
    test "$(git rev-parse HEAD)" = "${PLUGIN_SHA}"; \
    python3 -m pip install --no-cache-dir --no-deps -e .; \
    python3 -c "import sgl_amd_mono; assert 'Qwen3_5MoeForCausalLM' in {k[0] if isinstance(k, tuple) else k for k in sgl_amd_mono.SPECS}"; \
    python3 -c "import inspect; from sglang.kernels.plugin_kernels.amd.spec import register; assert 'model_type' in inspect.signature(register).parameters"; \
    grep -q '^def shared_down(' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/wide/stages/moe.py; \
    grep -q '^LAYER_WIDTHS_F0 = (3, 6, 8)$' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    mkdir -p /etc/sglang-build; \
    { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "sglang_base=${SGLANG_REV}"; \
      echo "aiter=${AITER_REV}"; \
      echo "sgl_amd_mono=${PLUGIN_SHA}"; \
    } >> /etc/sglang-build/manifest.txt; \
    cat /etc/sglang-build/manifest.txt; \
    python3 -m pip list --format=freeze > /sgl-workspace/pip_freeze.txt
