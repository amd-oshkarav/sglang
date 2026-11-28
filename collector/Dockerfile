# collector_261128_qwen_3.8_mono_attn: collector_261128_qwen_3.8_mono_og (the 20261127 SGLang nightly with the
# sgl-amd-mono plugin at PLUGIN_SHA of omirosh/sgl-amd-mono, q38v3/20-og) plus two SGLang PRs for the attention front:
# - sgl-project/sglang#47014 at f010ef9843: the decode KV write also builds the split-KV plan of vattn_asm, and
#   cu_seqlens_q reuses qo_indptr_unified_decode.
# - sgl-project/sglang#47779 at 83d026e84d (with its base sgl-project/sglang#45529): on aiter, one Triton kernel does
#   the Q/K GemmaRMSNorm, the partial RoPE and the gate split, with one wavefront per head and the gate read in place.
# SGLANG_REV is the second [Port] commit on this branch, on SGLANG_BASE of the base image. The PRs change Python and
# Triton files only: the build applies the diff SGLANG_BASE..SGLANG_REV to the editable SGLang tree of the base, which
# keeps the ROCm pyproject files of the nightly build, and checks that the 7 files are SGLANG_REV's.
# No AITER change. SGLANG_AMD_MONO_K1V=0 turns K1v off.
# Build: docker build --network=host -t amdsiloai/sglang-private:collector_261128_qwen_3.8_mono_attn collector/
ARG BASE_IMAGE=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261127@sha256:235b2a4f83ac071b081324bb338dc87210263f997448666ad489141c3594d777
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SGLANG_BASE=102a17d4df19be4941d9e5ebb73418b00a22e347
ARG SGLANG_REPO=https://github.com/amd-oshkarav/sglang.git
ARG SGLANG_REV=5059e52bd890c7a0ea10df084538c294f1ca0e3b
ARG PR47014_HEAD=f010ef98431a1f6ce85a565ca6027fd696dec742
ARG PR47779_HEAD=83d026e84d8b0fc6310c4e9f1c562257383d82bb
ARG AITER_REV=581579ab51ca26ddd4a08a6aa6aa983a78aaf276
ARG PLUGIN_REPO=https://github.com/omirosh/sgl-amd-mono.git
ARG PLUGIN_SHA=3cf33b156d60dc6c86d90e160e84453bd69d7b62
LABEL ai.amd.base_image="${BASE_IMAGE}" ai.amd.sglang="${SGLANG_REPO}@${SGLANG_REV}" ai.amd.aiter="${AITER_REV}" \
      ai.amd.sglang_pr47014="${PR47014_HEAD}" ai.amd.sglang_pr47779="${PR47779_HEAD}" \
      ai.amd.sgl_amd_mono="${PLUGIN_REPO}@${PLUGIN_SHA}"

RUN set -eux; \
    git config --global --add safe.directory '*'; \
    test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_BASE}"; \
    git -C /sgl-workspace/sglang fetch "${SGLANG_REPO}" "${SGLANG_REV}"; \
    test "$(git -C /sgl-workspace/sglang rev-parse FETCH_HEAD)" = "${SGLANG_REV}"; \
    test "$(git -C /sgl-workspace/sglang rev-parse "${SGLANG_REV}~2")" = "${SGLANG_BASE}"; \
    files="python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py python/sglang/kernels/ops/attention/unified_attention_3d_mtp.py python/sglang/kernels/ops/attention/vattn_asm_gfx950/__init__.py python/sglang/kernels/ops/kvcache/cache_ops.py python/sglang/srt/layers/attention/aiter_backend.py python/sglang/srt/models/qwen3_5.py test/registered/kernels/ops/attention/test_fused_qk_rmsnorm_rope_gate.py"; \
    test "$(git -C /sgl-workspace/sglang diff --name-only "${SGLANG_BASE}" "${SGLANG_REV}" | sort | tr '\n' ' ')" = "${files} "; \
    git -C /sgl-workspace/sglang diff --quiet "${SGLANG_BASE}" -- ${files}; \
    git -C /sgl-workspace/sglang diff "${SGLANG_BASE}" "${SGLANG_REV}" | git -C /sgl-workspace/sglang apply; \
    git -C /sgl-workspace/sglang diff --quiet "${SGLANG_REV}" -- ${files}; \
    git -C /sgl-workspace/sglang status --short; \
    (cd / && python3 -c "import sglang.srt.layers.attention.aiter_backend as m; assert m.__file__.startswith('/sgl-workspace/sglang/python/'), m.__file__"); \
    (cd / && python3 -c "from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate as f; import inspect; assert 'copy_gate' in inspect.signature(f).parameters"); \
    grep -q '^def alloc_decode_seg_plan(' /sgl-workspace/sglang/python/sglang/kernels/ops/attention/unified_attention_3d_mtp.py; \
    grep -q '^def seg_plan_program(' /sgl-workspace/sglang/python/sglang/kernels/ops/kvcache/cache_ops.py; \
    grep -q 'seg_plan=seg_plan,' /sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py; \
    grep -q 'copy_gate=False,' /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py; \
    grep -q '"num_warps": 1} if torch.version.hip' /sgl-workspace/sglang/python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py; \
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
    grep -q '^LAYER_WIDTHS_F0 = (2, 3, 6, 8)$' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    grep -q '^def _gate_input(gates: list):' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    grep -q 'keys.append(W.LayerBuild(kind="attn", og=True, \*\*common))' sgl_amd_mono/qwen38_a95b_mxfp4/spec.py; \
    mkdir -p /etc/sglang-build; \
    { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "sglang_base=${SGLANG_BASE}"; \
      echo "sglang=${SGLANG_REPO}@${SGLANG_REV}"; \
      echo "sglang_pr47014=${PR47014_HEAD}"; \
      echo "sglang_pr47779=${PR47779_HEAD}"; \
      echo "aiter=${AITER_REV}"; \
      echo "sgl_amd_mono=${PLUGIN_SHA}"; \
    } >> /etc/sglang-build/manifest.txt; \
    cat /etc/sglang-build/manifest.txt; \
    python3 -m pip list --format=freeze > /sgl-workspace/pip_freeze.txt
