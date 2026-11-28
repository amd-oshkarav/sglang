# collector_261128_qwen_3.8_mono_og: the 20261127 SGLang nightly with the sgl-amd-mono plugin at
# PLUGIN_SHA of omirosh/sgl-amd-mono: q38v3/20-og: q38v3/19-all (the image collector_261128_qwen_3.8_mono_all) plus
# the attention output gate in the launch after attention (og): K2, layer_block and the wide attn launch multiply
# SGLang's fused_sigmoid_mul gate into o_proj's input, so the attention layers run no _fused_sigmoid_mul_kernel.
# SGLANG_AMD_MONO_K1V=0 turns K1v off.
# No SGLang change and no AITER change: the build checks the SGLang and AITER commits of the base.
# Build: docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_og collector/
ARG BASE_IMAGE=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SGLANG_REV=102a17d4df19be4941d9e5ebb73418b00a22e347
ARG AITER_REV=581579ab51ca26ddd4a08a6aa6aa983a78aaf276
ARG PLUGIN_REPO=https://github.com/omirosh/sgl-amd-mono.git
ARG PLUGIN_SHA=c7e169eec6a56a89588d7e25cb6e7beeb5e72acb
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
    grep -q '^class Ring(NamedTuple):' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/gdn.py; \
    grep -q '^def _finish_gdn_replayssm_circular_fold_kernel(' /sgl-workspace/sglang/python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py; \
    grep -q '^UP_MAX = 3$' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/mono/blocks.py; \
    grep -q 'return "mxfp4_h32" if rotated' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    grep -q '^def shared_down(' sgl_amd_mono/qwen38_a95b_mxfp4/gfx950/wide/stages/moe.py; \
    grep -q '^LAYER_WIDTHS_F0 = (3, 6, 8)$' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    grep -q '^def _gate_input(gates: list):' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    grep -q 'keys.append(W.LayerBuild(kind="attn", og=True, \*\*common))' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    mkdir -p /etc/sglang-build; \
    { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "sglang_base=${SGLANG_REV}"; \
      echo "aiter=${AITER_REV}"; \
      echo "sgl_amd_mono=${PLUGIN_SHA}"; \
    } >> /etc/sglang-build/manifest.txt; \
    cat /etc/sglang-build/manifest.txt; \
    python3 -m pip list --format=freeze > /sgl-workspace/pip_freeze.txt
