#!/usr/bin/env bash
# AITER at ${AITER_REV} plus aiter#8829 (/tmp/aiter_pr8829.patch). No other patch, no table copy, no edit.
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
git apply --check /tmp/aiter_pr8829.patch
git apply /tmp/aiter_pr8829.patch
grep -q 'route_rows_switch' aiter/fused_moe.py
grep -q 'AITER_MXFP4_INTERMEDIATE' aiter/aot/flydsl/moe.py
echo "aiter#8829 applied at ${AITER_PR8829_HEAD} on ${got}"
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
# The AITER functions that the SGLang of this branch imports on the Qwen3.8 path (FlyDSL
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
echo "aiter ${got} plus aiter#8829, Q256 rows ${q256n}"
