[example parent sweep](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/e9dff90720f34ad49bc50058cc210d97?o=4032599967441609)

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

Base: `ROCm/aiter` main @ `90596736129ccc0bcf720686cf202067a11f110d` (2026-10-26). This is the AITER of the branch
`collector_09_11_qwen_3.8_stack_p2`. It is older than the merges of aiter#5988 (the Q256 MoE prefill kernel) and
aiter#7441 (the Qwen3.8 a4w4 GEMM table), because the later versions introduce ~<> 2-3% rise of TTFT measured w.r.t  commit `905...` on the same nodes. One of the sources of this regression is aiter#5988 (the Q256 MoE prefill kernel), the possible others are being investigated. 

Other AITER pins used for files only:

| Pin | Role |
| --- | --- |
| `ec91f61f0c550811cf121f0b0dc667c09524b161` (`AITER_G1_REV`) | merge commit of aiter#7441, source of the a4w4 GEMM table |
| `cf16a5528f8fcce63c427987bfac30f52741e803` (`AITER_TABLES_REV`) | AITER of the collector_24_11 runs, source of the two MoE tables and the ASM kernels |

### 2.1 Table added (not in 90596736 at all)

| File | Source | Content |
| --- | --- | --- |
| `aiter/configs/model_configs/qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv` | copied from `ec91f61f` (aiter#7441) | 26 tuned a4w4 blockscale GEMM rows, N=4608, K=8192 (the "G-1" table) |

The untuned table of aiter#7441 is not copied. The same file is unchanged at `cf16a552`.

### 2.2 Tables ported from cf16a552 and edited (they exist in 90596736 and are replaced)

Both files are downloaded from `cf16a552` and written over the 90596736 versions, then `aiter_no5988.patch` is applied:

| File | After the port and the edit, compared with the 90596736 file |
| --- | --- |
| `qwen3_5_397b_fp4_tuned_fmoe.csv` | 7 more rows (1 to 64 tokens). The 3 Q256 rows (8192, 16384, 32768 tokens) that `cf16a552` has are removed. |
| `qwen3_8_2400b_fp4_tuned_fmoe.csv` | 4 changed rows (32 and 64 tokens, E=512 and E=513). The 3 Q256 rows are replaced by FlyDSL rows at 8192, 16384 and 32768 tokens (`flydsl_moe1_..._fp4` plus `flydsl_moe2_layout_..._reduce_...`). |

The "no5988" in the image name means these Q256 rows (aiter#5988) are taken out again. The kernel
`fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route` is not used.
The image build belowasserts 3 Q256 rows before the patch and 0 after, and 3 `flydsl_moe1_` rows at 8192, 16384 and 32768 in the qwen3_8 table.

### 2.3 ASM MoE port (needed by the ported tables)

Four rows of the ported qwen3_8 table name the ASM kernel `fmoe_bf16_pertokenMXfp4_g1u1_emsort_vs_silu_16xstk`,
which 90596736 does not have. The three `flat_*` kernels the tables use are older builds in 90596736 than in cf16a552.
So `aiter_setup.sh` also brings over from cf16a552:

| Item | What is done |
| --- | --- |
| Kernel binaries | 4 `.co` files into `hsa/gfx950/fmoe/silu/`: `emsort_vs_silu_16xstk` (new), `flat_novs_silu_16x128`, `flat_novs_silu_16x256`, `flat_vs_silu_16x32` (replaced). The build checks the ELF magic. |
| Manifest | `fmoe_bf16_pertokenMXfp4_g1u1_silu.csv` keeps its 11 rows and gains the emsort row, in the new schema with `max_tokens`, `max_assignments`, `max_experts` (emsort row: flat=2, 64, 2048, 1024). |
| Launcher | `aiter_fmoe_asm_port.patch`: `csrc/py_itfs_cu/asm_fmoe.cu` reads the EMSORT limits from the manifest instead of the hard-coded 512 assignments and 1024 experts, and checks that the grid is resident. `aiter/fused_moe.py` returns an empty `num_valid_ids` tensor (null pointer) in `_moe_prepare_unsorted_input`, as cf16a552 does. |

The build runs a check that every ASM kernel named in the two tables has a `.co` and a manifest row.

Not taken from cf16a552: the Q256 prefill rows, the POST_REDUCE mode, the new kernel selection heuristic.
Also not applied: the P-1b edit (`if force_reduce or fp8_reduce:` stays). The environment variable `AITER_FLYDSL_STAGE2_FP8=1` is set in the config.

Caveat from the Dockerfile (see below) header: the C++ port and the ported kernels were not run where they were written. The first run should
show the concurrency 32 and 64 cells (they use the emsort row) before results are trusted.

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

## 4. How the image is built

Steps of the `Dockerfile`, in order:

1. `FROM lmsysorg/sglang:v0.5.23-rocm10-mi35x`. Triton of the base is compared with `EXPECT_TRITON` and the build stops on a mismatch.
2. Rust (cargo) is installed if missing.
3. SGLang: the old `sglang` and `sgl-kernel` are uninstalled, `collector_24_11` is cloned blobless (so setuptools-scm sees the tags), checked out at `SGLANG_REV`, submodules are updated.
4. `sgl-kernel` (or `python/sglang/kernels/aot`) is built with `setup_rocm.py install`.
5. `sglang[all_hip]` (torch 2.9 or 2.10) or `sglang[all_hip_rocm724]` is installed editable, with `torch` and `triton` pinned through `/tmp/constraints.txt`, plus `sgl-eval==0.1.2`. The build fails if CUDA runtime packages appear.
6. `COPY` of `aiter_setup.sh` and the two patches, then `RUN bash /tmp/aiter_setup.sh` (section 2). It uninstalls `amd-aiter`, clones AITER at `AITER_REV`, copies and ports the tables and ASM files, applies the two patches, rebuilds AITER with `AITER_USE_SYSTEM_TRITON=1 AITER_REBUILD=1 python3 setup.py develop`, and checks that the Triton packages did not change and that the names SGLang imports exist.
7. A check that the SGLang tree is still the pinned commit and has the FlyDSL GDN prefill files.
8. sgl-amd-mono: clone, checkout, cherry-pick PR 154 (section 3), editable install, entry point check.
9. `LABEL`s record all pins (`ai.amd.aiter.commit`, `ai.amd.aiter.fmoe_tables`, `ai.amd.sgl_amd_mono.pr`, ...).

Manual build (from the repository root):

```
docker build --network=host --progress=plain \
    -f docker/collector_24_11_aiter_week47_cf16tables/Dockerfile \
    --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x \
    -t amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_24_11_c4a2edc9_aiter_90596736_g1_cf16fmoe_asmport_no5988_mono_4cba5b25_pr154 \
    docker/collector_24_11_aiter_week47_cf16tables
```

Run with ITT from the repository root:

```
inference-testing -c docs/examples/collector_24_11/week47_aiter/qwen_3.8_collector_24_11_aiter_90596736_cf16_fmoe_rr0_8.yaml
```

Server environment set in the config (not in the image): `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `SGLANG_ROCM_USE_AITER_TOPK_GATING=1`,
`SGLANG_ROCM_GDN_ONLINE_MXFP4=1`, `AITER_FLYDSL_STAGE2_FP8=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`,
`ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`.
Server flags include `--tp-size 8`, `--attention-backend aiter`, `--linear-attn-prefill-backend flydsl`, `--kv-cache-dtype fp8_e4m3`,
`--mamba-ssm-dtype bfloat16`, `--page-size 16`, `--enable-amd-monokernel`, `--enable-aiter-allreduce-fusion`.

## 5. Files (verbatim)

### 5.1 `docker/collector_24_11_aiter_week47_cf16tables/Dockerfile`

```dockerfile
# Feature image for the collector_24_11 branch with the AITER of the
# collector_09_11_qwen_3.8_stack_p2 branch (week 47) and two fmoe tune tables of
# the collector_24_11 AITER. Built from the stock base,
# three layers on lmsysorg/sglang:v0.5.23-rocm10-mi35x. It does not need any
# earlier collector image, and docker/collector_24_11/Dockerfile is unchanged.
#
# 1. SGLang. amd-oshkarav/sglang collector_24_11 @ c4a2edc924. The branch is
#    sgl-project main 88dc7501f5 plus seven features that main does not have
#    yet: #39069, #40771, FlyDSL INT4 QuickReduce (8cb39392), #47219, #47433,
#    #45529 and #39595 (the FlyDSL GDN prefill backend,
#    --linear-attn-prefill-backend flydsl). The tune-table all-reduce (#50069)
#    is not on this branch. The checkout is cloned and reinstalled, and nothing
#    in the tree is patched.
#
# 2. AITER. ROCm/aiter main @ 90596736 (2026-10-26), the AITER of the branch
#    collector_09_11_qwen_3.8_stack_p2 (its README, section "AITER"), and the
#    same recipe as docker/collector_19_11_mono_aiter_week47 (aiter_setup.sh):
#    - The a4w4 GEMM tuned table of aiter#7441 (qwen38_2p4t_a4w4_blockscale_
#      tuned_gemm.csv at ec91f61f) is copied in. 90596736 is older than the
#      merge, so the file is not there. The untuned table is not copied.
#    - No P-1b edit. The branch README applies it but says it has no effect on
#      Qwen3.8, and the week 47 image does not have it.
#    - aiter#5988 is not in 90596736 (it is older than 0e072ca50e), so this
#      AITER has no Q256 rows. The build checks that before the next step.
#    - The two fmoe tune tables that aiter_no5988.patch names,
#      qwen3_5_397b_fp4_tuned_fmoe.csv and qwen3_8_2400b_fp4_tuned_fmoe.csv, are
#      taken from AITER cf16a552 (AITER_TABLES_REV), the AITER of the
#      collector_24_11 runs, and put over the tables of 90596736. cf16a552 has
#      the Q256 rows of aiter#5988, so aiter_no5988.patch (the same file as in
#      docker/collector_24_11) is then applied to them: it restores the FlyDSL
#      rows at 8192, 16384 and 32768 tokens in the qwen3_8 table and drops the
#      three Q256 rows from the qwen3_5 table. Against the tables of 90596736
#      the result has 7 more rows in the qwen3_5 table (1 to 64 tokens) and 4
#      changed rows in the qwen3_8 table (32 and 64 tokens, E=512 and E=513).
#      Those 4 rows name the ASM kernel
#      fmoe_bf16_pertokenMXfp4_g1u1_emsort_vs_silu_16xstk, which 90596736 does
#      not have, so the ASM MoE part is ported too (aiter_setup.sh, "asmport"):
#      - the .co of that kernel, and the .co of the three flat_* kernels the
#        tables use (flat_novs_silu_16x128, flat_novs_silu_16x256,
#        flat_vs_silu_16x32), which are other builds in cf16a552 than in
#        90596736, are copied from cf16a552;
#      - the manifest hsa/gfx950/fmoe/silu/fmoe_bf16_pertokenMXfp4_g1u1_silu.csv
#        keeps its 11 rows and gains the emsort row, in the new schema with
#        max_tokens, max_assignments and max_experts;
#      - aiter_fmoe_asm_port.patch ports the EMSORT manifest limits of
#        csrc/py_itfs_cu/asm_fmoe.cu (90596736 stops an EMSORT launch at 512
#        assignments, the emsort row has 2048) and the null num_valid_ids of
#        aiter/fused_moe.py (the new FLAT and EMSORT binaries read it as the
#        token count when the pointer is set). It is the cf16a552 code, cut
#        down to those parts.
#      Not taken: the Q256 prefill rows, the POST_REDUCE mode and the new
#      kernel selection heuristic. Tuned rows name their kernel, so the
#      heuristic does not decide them. Neither the C++ nor the kernels could be
#      run where this was written, so the first run has to prove them: the
#      concurrency 32 and 64 cells use the emsort row. The build stops when a
#      tuned row names an ASM kernel with no .co or no manifest row.
#    - AITER_USE_SYSTEM_TRITON=1 keeps the Triton of the base image. The build
#      compares the Triton packages before and after the AITER step and fails
#      if they changed. EXPECT_TRITON stops the build early when the base has
#      another Triton (set it to an empty string to skip the check). Unlike
#      cf16a552 and newer main, 90596736 has no gl.amd.cdna5 line in
#      aiter/ops/triton/_gluon_kernels/common/utils.py.
#    - This AITER has no FlyQuickAllReduce. aiter.ops.flydsl exports
#      QuickAllReduceInt4, the only AITER all-reduce name the collector_24_11
#      diff imports. The build checks, in the AITER source, the functions that
#      the FlyDSL GDN prefill backend imports, and imports QuickAllReduceInt4
#      and GateMode.
#
# 3. sgl-amd-mono. main @ 4cba5b25, then the commits of the PRs in
#    SGL_AMD_MONO_PRS cherry-picked on top, in that order. The default is PR
#    154, five commits up to its head 05b9dd9f:
#    https://github.com/sgl-project/sgl-amd-mono/pull/154. A PR is taken from
#    the merge base with SGL_AMD_MONO_REV up to the head named in
#    SGL_AMD_MONO_PR_REVS ("154=<sha>"), or up to the head of the PR at build
#    time when it has no entry there. PR 147 is not in the default list. Its
#    change is in main since 36c385e.
#
#   docker build --network=host --progress=plain \
#       -f docker/collector_24_11_aiter_week47_cf16tables/Dockerfile \
#       --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.23-rocm10-mi35x \
#       -t amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_24_11_c4a2edc9_aiter_90596736_g1_cf16fmoe_asmport_no5988_mono_4cba5b25_pr154 \
#       docker/collector_24_11_aiter_week47_cf16tables
#
# ITT needs dockerfile.path to be this file:
#   /app/repos/inference-testing/docker/collector_24_11_aiter_week47_cf16tables/Dockerfile

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
ARG AITER_REV=90596736129ccc0bcf720686cf202067a11f110d
# The merge commit of aiter#7441. Only the tuned a4w4 table is taken from it.
ARG AITER_G1_REV=ec91f61f0c550811cf121f0b0dc667c09524b161
# The AITER whose two fmoe tune tables are used (the collector_24_11 AITER).
ARG AITER_TABLES_REV=cf16a5528f8fcce63c427987bfac30f52741e803
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
        echo "ERROR: the base image has triton ${got:-none}, the week 47 image has ${EXPECT_TRITON}." >&2; \
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

# AITER 90596736 plus the aiter#7441 tuned a4w4 table, as in the week 47 image,
# plus the two fmoe tune tables of AITER_TABLES_REV edited by aiter_no5988.patch,
# plus the ASM MoE kernels, manifest rows and launcher code those tables need.
# The step is in aiter_setup.sh and is copied in: ITT's classic builder does not
# feed a RUN heredoc to the command.
COPY aiter_setup.sh /tmp/aiter_setup.sh
COPY aiter_no5988.patch /tmp/aiter_no5988.patch
COPY aiter_fmoe_asm_port.patch /tmp/aiter_fmoe_asm_port.patch
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
      ai.amd.aiter.g1="${AITER_G1_REV}" \
      ai.amd.aiter.p1b="none" \
      ai.amd.aiter.fmoe_tables="${AITER_TABLES_REV}" \
      ai.amd.aiter.fmoe_tables_edit="aiter_no5988.patch" \
      ai.amd.aiter.fmoe_asm_port="aiter_fmoe_asm_port.patch" \
      ai.amd.aiter.use_system_triton="${AITER_USE_SYSTEM_TRITON}" \
      ai.amd.sgl_amd_mono.repo="${SGL_AMD_MONO_REPO}" \
      ai.amd.sgl_amd_mono.commit="${SGL_AMD_MONO_REV}" \
      ai.amd.sgl_amd_mono.pr="${SGL_AMD_MONO_PRS}" \
      ai.amd.sgl_amd_mono.pr_revs="${SGL_AMD_MONO_PR_REVS}" \
      ai.amd.sglang.arm="collector-24-11-aiter-week47-cf16-fmoe-asmport"

WORKDIR /sgl-workspace/sglang
```

### 5.2 `docker/collector_24_11_aiter_week47_cf16tables/aiter_setup.sh`

```bash
#!/usr/bin/env bash
# AITER 90596736 plus the aiter#7441 tuned a4w4 table, as in the week 47 image,
# plus the two fmoe tune tables of ${AITER_TABLES_REV} edited by aiter_no5988.patch,
# plus the ASM MoE kernels, manifest rows and launcher code those tables need.
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
# aiter#5988 is not in this commit. If a Q256 row is here, the commit is wrong.
if grep -q 'fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1' "${fmoe}"; then
    echo "ERROR: Q256 MoE rows are present. ${got} is past aiter#5988." >&2
    exit 1
fi
# No P-1b edit: the line must be the original one.
moe="aiter/ops/flydsl/moe_kernels.py"
if ! grep -q 'if force_reduce or fp8_reduce:' "${moe}"; then
    echo "ERROR: ${moe} has no 'if force_reduce or fp8_reduce:' at ${got}. This image has no P-1b edit." >&2
    exit 1
fi
if ! grep -q 'def chunk_gated_delta_rule_fwd_h_flydsl_opt' aiter/ops/flydsl/linear_attention_prefill_kernels.py; then
    echo "ERROR: ${got} has no chunk_gated_delta_rule_fwd_h_flydsl_opt" >&2
    exit 1
fi
# aiter#7441 is not in this commit, so neither a4w4 table is there. Only the
# tuned table is copied, from the merge commit.
tuned=qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv
for name in "${tuned}" qwen38_2p4t_a4w4_blockscale_untuned_gemm.csv; do
    if [ -e "${cfg}/${name}" ]; then
        echo "ERROR: ${name} already exists at ${got}" >&2
        exit 1
    fi
done
curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors \
    -fsSL \
    -o "${cfg}/${tuned}" \
    "https://raw.githubusercontent.com/ROCm/aiter/${AITER_G1_REV}/aiter/configs/model_configs/${tuned}"
test -s "${cfg}/${tuned}"
# The two fmoe tune tables that aiter_no5988.patch names, from AITER
# ${AITER_TABLES_REV}, over the tables of this AITER. That AITER has the Q256 rows
# of aiter#5988, so the patch is applied to them, as in docker/collector_24_11.
for t in qwen3_5_397b_fp4_tuned_fmoe.csv qwen3_8_2400b_fp4_tuned_fmoe.csv; do
    curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors \
        -fsSL \
        -o "${cfg}/${t}" \
        "https://raw.githubusercontent.com/ROCm/aiter/${AITER_TABLES_REV}/${cfg}/${t}"
    test -s "${cfg}/${t}"
done
q256=prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route
qwen38="${cfg}/qwen3_8_2400b_fp4_tuned_fmoe.csv"
qwen35="${cfg}/qwen3_5_397b_fp4_tuned_fmoe.csv"
test "$(grep -c "${q256}" "${qwen38}")" = 3
test "$(grep -c "${q256}" "${qwen35}")" = 3
git apply --check /tmp/aiter_no5988.patch
git apply /tmp/aiter_no5988.patch
test "$(grep -c "${q256}" "${qwen38}")" = 0
test "$(grep -c "${q256}" "${qwen35}")" = 0
test "$(grep -c '^256,\(8192\|16384\|32768\),8192,256,513,.*flydsl_moe1_' "${qwen38}")" = 3
sha256sum "${qwen38}" "${qwen35}"
# The tuned rows name pertokenMXfp4 ASM kernels. Four rows of the qwen3_8 table
# name fmoe_bf16_pertokenMXfp4_g1u1_emsort_vs_silu_16xstk, which 90596736 lacks,
# and the three flat_* kernels that the tables use are older builds here than in
# ${AITER_TABLES_REV}, where the tables were tuned. So the binaries, the manifest
# and the launcher are brought over from ${AITER_TABLES_REV} too:
#   1. the four .co files below go over hsa/gfx950/fmoe/silu/;
#   2. the manifest keeps the 11 rows of this AITER and gains the emsort row, in
#      the new schema (max_tokens, max_assignments, max_experts; hsa/codegen.py
#      turns every column into a field of the generated config);
#   3. aiter_fmoe_asm_port.patch ports the EMSORT manifest limits of
#      csrc/py_itfs_cu/asm_fmoe.cu (this AITER caps an EMSORT launch at 512
#      assignments, the emsort row needs 2048) and the null num_valid_ids of
#      aiter/fused_moe.py (the new FLAT and EMSORT binaries read it as the token
#      count when the pointer is set).
# Nothing else of ${AITER_TABLES_REV} is taken: no Q256 prefill rows, no POST_REDUCE
# mode, no change to the kernel selection heuristic.
silu=hsa/gfx950/fmoe/silu
reg="${silu}/fmoe_bf16_pertokenMXfp4_g1u1_silu.csv"
raw="https://raw.githubusercontent.com/ROCm/aiter/${AITER_TABLES_REV}"
test "$(wc -l < "${reg}")" = 12
for co in fmoe_bf16_pertokenMXfp4_g1u1_emsort_vs_silu_16xstk \
          fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x128 \
          fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256 \
          fmoe_bf16_pertokenMXfp4_g1u1_flat_vs_silu_16x32; do
    curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors \
        -fsSL -o "/tmp/${co}.co" "${raw}/${silu}/${co}.co"
    test -s "/tmp/${co}.co"
    # A .co is an ELF file.
    test "$(head -c 4 "/tmp/${co}.co" | od -An -tx1 | tr -d ' \n')" = 7f454c46
    if [ "${co}" = fmoe_bf16_pertokenMXfp4_g1u1_emsort_vs_silu_16xstk ]; then
        test ! -e "${silu}/${co}.co"
    else
        test -e "${silu}/${co}.co"
    fi
    cp "/tmp/${co}.co" "${silu}/${co}.co"
done
curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors \
    -fsSL -o /tmp/cf16_silu_manifest.csv "${raw}/${reg}"
REG="${reg}" python3 - <<'PY'
import csv
import os

reg = os.environ["REG"]
old = {r["knl_name"] for r in csv.DictReader(open(reg))}
emsort = "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_emsort_vs_silu_16xstkE"
src = csv.DictReader(open("/tmp/cf16_silu_manifest.csv"))
fields = src.fieldnames
assert fields[-3:] == ["max_tokens", "max_assignments", "max_experts"], fields
keep = [r for r in src if r["knl_name"] in old or r["knl_name"] == emsort]
assert len(old) == 11 and len(keep) == 12, (len(old), len(keep))
assert {r["knl_name"] for r in keep} == old | {emsort}
em = next(r for r in keep if r["knl_name"] == emsort)
assert (em["flat"], em["max_tokens"], em["max_assignments"], em["max_experts"]) == ("2", "64", "2048", "1024"), em
with open(reg, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(keep)
print("manifest rows:", len(keep), "(11 of this AITER + the emsort row)")
PY
git apply --check /tmp/aiter_fmoe_asm_port.patch
git apply /tmp/aiter_fmoe_asm_port.patch
grep -q 'max_assignments' csrc/py_itfs_cu/asm_fmoe.cu
grep -q 'num_valid_ids = torch.empty(0, dtype=dtypes.i32, device=device)' aiter/fused_moe.py
sha256sum "${reg}" "${silu}"/fmoe_bf16_pertokenMXfp4_g1u1_{emsort_vs_silu_16xstk,flat_novs_silu_16x128,flat_novs_silu_16x256,flat_vs_silu_16x32}.co
# A row whose kernel is an ASM symbol (_ZN5aiter<len><name>E) needs <name>.co
# under hsa/ and a row of the manifest. A row that names one this AITER lacks is
# an error: the server would fail on that shape at run time.
python3 - <<'PY'
import csv
import pathlib
import re
import sys

cfg = pathlib.Path("aiter/configs/model_configs")
have = {p.stem for p in pathlib.Path("hsa").rglob("*.co")}
manifest = {
    r["knl_name"]
    for p in pathlib.Path("hsa").rglob("*.csv")
    for r in csv.DictReader(open(p))
    if "knl_name" in r
}
missing = set()
for t in ("qwen3_5_397b_fp4_tuned_fmoe.csv", "qwen3_8_2400b_fp4_tuned_fmoe.csv"):
    for row in csv.DictReader(open(cfg / t)):
        for key in ("kernelName1", "kernelName2"):
            sym = row.get(key) or ""
            m = re.fullmatch(r"_ZN5aiter(\d+)(.+)E", sym)
            if not m:
                continue
            name = m.group(2)[: int(m.group(1))]
            if name not in have or sym not in manifest:
                missing.add((t, name))
for t, name in sorted(missing):
    print(f"ERROR: {t} names the ASM kernel {name}, which has no .co or no manifest row")
print("ASM kernel check:", "all present" if not missing else f"{len(missing)} missing")
sys.exit(1 if missing else 0)
PY
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
# The all-reduce class that the tune tables name, and the MoE GateMode that
# SGLang imports, must import. The optional names are only reported.
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
echo "aiter ${got}, P-1b none, aiter#5988 not in the commit, a4w4 tuned table from ${AITER_G1_REV}, fmoe tables from ${AITER_TABLES_REV} edited by aiter_no5988.patch, ASM MoE port (emsort + flat .co, manifest, launcher) from ${AITER_TABLES_REV}"
```

### 5.3 `docker/collector_24_11_aiter_week47_cf16tables/aiter_no5988.patch`

```diff
diff --git a/aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv b/aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv
index 057b46f9..b47cebd0 100644
--- a/aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv
+++ b/aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv
@@ -54,6 +54,3 @@ gfx950,256,8,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m
 gfx950,256,16,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,49.7641,flydsl_mxmoe_g1_a4w4_16x128x256_f16in_nt,1.1%,0,flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16,1.1%,49.7641,0,0,0,22.25,32432.12,
 gfx950,256,32,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,73.7098,flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt,1.1%,0,flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16,1.1%,73.7098,0,0,0,30.04,21898.74,
 gfx950,256,64,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,16,0,113.601,flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt,1.1%,0,flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16,1.1%,113.601,0,0,0,38.99,14212.42,
-gfx950,256,8192,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,256,0,558.6791,fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route,,0,,,558.6791,1,0,3,,,
-gfx950,256,16384,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,256,0,956.1762,fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route,,0,,,956.1762,1,0,3,,,
-gfx950,256,32768,4096,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,256,0,1615.8382,fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route,,0,,,1615.8382,1,0,3,,,
diff --git a/aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv b/aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv
index 15348f8e..397cc36b 100644
--- a/aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv
+++ b/aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv
@@ -28,6 +28,6 @@ cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,
 256,1024,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,183.6767,flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w4_kw2_fp4,1.0%,159.1087,flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_reduce_nt_sbm32,1.2%,342.7854,0,0,0,413.48,9488.98,
 256,2048,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,204.474,flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w3_fp4,1.0%,230.2579,flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_nt_sbm64,1.2%,434.7319,0,0,0,652.05,7539.93,
 256,4096,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,128,0,239.8522,flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w3_fp4,1.0%,397.2772,flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_nt_sbm128,1.2%,637.1294,0,0,0,889.83,5223.71,
-256,8192,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,256,0,1117.2118,fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route_weighted,,0,,,1117.2118,1,0,3,,,
-256,16384,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,256,0,1864.0202,fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route,,0,,,1864.0202,1,0,3,,,
-256,32768,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,256,0,3176.0620,fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route,,0,,,3176.0620,1,0,3,,,
+256,8192,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,352.8917,flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_bnt0_xcd4_fp4,1.0%,696.5154,flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64,1.2%,1049.4071,0,0,0,1080.49,3267.41,
+256,16384,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,128,0,543.2484,flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w3_bnt0_xcd4_fp4,1.0%,1361.4173,flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm128,1.2%,1904.6657,0,0,0,1190.63,1905.94,
+256,32768,8192,256,513,11,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,64,0,1075.9122,flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_bnt0_xcd4_fp4,0.1%,3236.5104,flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_persist_sbm64,0.1%,4312.4226,0,0,0,1051.73,935.17,
```

### 5.4 `docker/collector_24_11_aiter_week47_cf16tables/aiter_fmoe_asm_port.patch`

```diff
diff --git a/aiter/fused_moe.py b/aiter/fused_moe.py
index 4cf303c..d9b11db 100644
--- a/aiter/fused_moe.py
+++ b/aiter/fused_moe.py
@@ -293,9 +293,13 @@ def _moe_prepare_unsorted_input(topk_ids, topk_weights, model_dim, moebuf_dtype)
         if topk_weights.dtype == dtypes.fp32 and topk_weights.is_contiguous()
         else topk_weights.to(dtypes.fp32).contiguous()
     )
-    # sorted_expert_ids / num_valid_ids slots are unread by FLAT kernels,
-    # but must be valid device pointers -- alias topk_ids as scratch.
-    return topk_ids_i32, topk_weights_f32, topk_ids_i32, topk_ids_i32, moe_buf
+    # sorted_expert_ids is unread; alias topk_ids so the pointer is valid.
+    # The FLAT and EMSORT binaries of aiter cf16a552 read XC (num_valid_ids) as
+    # the token count whenever the pointer is set. An empty tensor has a null
+    # data pointer, so the kernel keeps the kernarg token_cnt, which is M. Ported
+    # from cf16a552, where this is the num_local_tokens=None case.
+    num_valid_ids = torch.empty(0, dtype=dtypes.i32, device=device)
+    return topk_ids_i32, topk_weights_f32, topk_ids_i32, num_valid_ids, moe_buf
 
 
 def _adaptive_moe_sort(
diff --git a/csrc/py_itfs_cu/asm_fmoe.cu b/csrc/py_itfs_cu/asm_fmoe.cu
index 4e958fd..2e670e7 100755
--- a/csrc/py_itfs_cu/asm_fmoe.cu
+++ b/csrc/py_itfs_cu/asm_fmoe.cu
@@ -80,6 +80,11 @@ class FMoeKernel
     const char* name            = nullptr;
     // flat_mode: 0 = host-sorted, 1 = flat one-token-per-TG grid, 2 = emsort persistent grid
     int flat_mode = 0;
+    // Manifest limits of an EMSORT binary (flat=2). 0 max_tokens means no separate
+    // token cap. Ported from aiter cf16a552.
+    uint32_t max_tokens      = 0;
+    uint32_t max_assignments = 0;
+    uint32_t max_experts     = 0;
 
     public:
     FMoeKernel(const char* name,
@@ -87,13 +92,19 @@ class FMoeKernel
                uint32_t sub_GU             = 512,
                uint32_t num_persistent_tgs = 0,
                int flat_mode               = 0,
-               uint32_t tgs_align          = 1) : kernel(name, hsaco)
+               uint32_t tgs_align          = 1,
+               uint32_t max_tokens         = 0,
+               uint32_t max_assignments    = 0,
+               uint32_t max_experts        = 0) : kernel(name, hsaco)
     {
         this->sub_GU             = sub_GU;
         this->num_persistent_tgs = num_persistent_tgs;
         this->tgs_align          = std::max(tgs_align, 1u);
         this->name               = name;
         this->flat_mode          = flat_mode;
+        this->max_tokens         = max_tokens;
+        this->max_assignments    = max_assignments;
+        this->max_experts        = max_experts;
     };
 
     const char* get_name() const { return name; }
@@ -207,13 +218,46 @@ class FMoeKernel
         }
         else if(this->flat_mode == 2)
         {
-            AITER_CHECK(static_cast<uint64_t>(token_cnt) * topk <= 512,
+            // Each EMSORT binary declares its own queue in the manifest
+            // (max_tokens, max_assignments, max_experts). A zero max_tokens
+            // means that binary has no separate token cap.
+            //
+            // The producer is the last Y workgroup. Every other workgroup spins
+            // on its ready flag, so this launch assumes the whole grid is
+            // resident together. num_persistent_tgs is tg_num_perCU * num_cu,
+            // rounded down to lcm(ps_deno, tgs_align), and the grid below is
+            // exactly that pool. Occupancy (whether tg_num_perCU fits in a CU)
+            // and room left by another stream or a CU mask are not queried.
+            AITER_CHECK(this->max_assignments > 0 && this->max_experts > 0,
                         __func__,
-                        ": EMSORT supports token_cnt*topk <= 512; got ",
+                        ": EMSORT manifest row must set max_assignments and max_experts for ",
+                        this->name);
+            if(this->max_tokens > 0)
+            {
+                AITER_CHECK(token_cnt <= static_cast<int>(this->max_tokens),
+                            __func__,
+                            ": EMSORT ",
+                            this->name,
+                            " supports token_cnt <= ",
+                            this->max_tokens,
+                            "; got ",
+                            token_cnt);
+            }
+            AITER_CHECK(static_cast<uint64_t>(token_cnt) * topk <= this->max_assignments,
+                        __func__,
+                        ": EMSORT ",
+                        this->name,
+                        " supports token_cnt*topk <= ",
+                        this->max_assignments,
+                        "; got ",
                         static_cast<uint64_t>(token_cnt) * topk);
-            AITER_CHECK(eprt <= 1024,
+            AITER_CHECK(eprt <= static_cast<int>(this->max_experts),
                         __func__,
-                        ": EMSORT supports at most 1024 experts; got ",
+                        ": EMSORT ",
+                        this->name,
+                        " supports at most ",
+                        this->max_experts,
+                        " experts; got ",
                         eprt);
             AITER_CHECK(inter_dim % sub_GU == 0,
                         __func__,
@@ -221,10 +265,14 @@ class FMoeKernel
                         inter_dim,
                         ", sub_GU=",
                         sub_GU);
+            AITER_CHECK(this->num_persistent_tgs > 0 && args.total_tgs > 0,
+                        __func__,
+                        ": EMSORT requires the dispatcher resident grid; got total_tgs=",
+                        args.total_tgs,
+                        ", pool=",
+                        this->num_persistent_tgs);
             gdx = inter_dim / sub_GU;
-            const int emsort_tgs = args.total_tgs > 0
-                                       ? static_cast<int>(args.total_tgs)
-                                       : 256;
+            const int emsort_tgs = static_cast<int>(args.total_tgs);
             AITER_CHECK(gdx > 0 && (emsort_tgs % gdx) == 0,
                         __func__,
                         ": EMSORT requires total_tgs divisible by inter_dim/sub_GU; got total_tgs=",
@@ -234,6 +282,17 @@ class FMoeKernel
             bdx = 256;
             gdy = emsort_tgs / gdx;
             gdz = 1;
+            const uint32_t launched =
+                static_cast<uint32_t>(gdx) * static_cast<uint32_t>(gdy);
+            // Documents the dispatcher assumption above. gdy is total_tgs / gdx
+            // and total_tgs is num_persistent_tgs rounded down, so launched is
+            // total_tgs and this comparison holds for every input that reached it.
+            AITER_CHECK(launched <= this->num_persistent_tgs,
+                        __func__,
+                        ": EMSORT grid must be fully resident; launched ",
+                        launched,
+                        " workgroups, dispatcher pool ",
+                        this->num_persistent_tgs);
         }
         else if(this->num_persistent_tgs != 0 && args.total_tgs > 0 &&
                 (args.total_tgs % args.ps_deno) == 0) // ps
@@ -394,8 +453,15 @@ FMoeKernel* get_heuristic_kernel(int inter_dim,
             num_persistent_tgs = 0;
 
         impl_ptr = &impl_ptr_map.get_or_create(name, [&]() {
-            return FMoeKernel(
-                name, co_name, cfg.subGU_n, num_persistent_tgs, cfg.flat, cfg.tgs_align);
+            return FMoeKernel(name,
+                              co_name,
+                              cfg.subGU_n,
+                              num_persistent_tgs,
+                              cfg.flat,
+                              cfg.tgs_align,
+                              static_cast<uint32_t>(std::max(cfg.max_tokens, 0)),
+                              static_cast<uint32_t>(std::max(cfg.max_assignments, 0)),
+                              static_cast<uint32_t>(std::max(cfg.max_experts, 0)));
         });
     }
     else
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
