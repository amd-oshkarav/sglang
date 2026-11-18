# Week 47 gdn_formats and lb_nofse, monokernel plugin

Image: `qwen38-gdnfmt:<sha7>-7397386`. The image is local only. Each job builds it on the node and does not push it to a registry. `<sha7>` is the sgl-amd-mono plugin commit of the arm.

This branch is SGLang `5bb7ecc2c2` (2026-10-28), the SGLang of the base image, plus one commit for [#46059](https://github.com/sgl-project/sglang/pull/46059). The image code is in `qwen38/spec_decode_mi355/mono_plugin/gdn_formats/` of [AMD-AGI/amd-silo-ai-tiger-teams-sglang](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/olga/qwen38/spec_decode_mi355/mono_plugin/gdn_formats) (branch `olga`).

The image has an AITER edit too. AITER is not in this repo. The section "AITER" gives the edit.

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-18) |
| --- | --- | --- | --- |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quark MXFP4 fused shared expert | Yes, through the base `5bb7ecc2c2` | Yes, merged 2026-10-26 |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | MXFP4 GDN projections, online | Yes, through the base `5bb7ecc2c2` | Yes, merged 2026-10-27 |
| [#46059](https://github.com/sgl-project/sglang/pull/46059) | Hadamard-32 rotation of the online MXFP4 GDN projections | `eb7811e22c`, the PR at `7397386194` | No, open |

- `eb7811e22c` applies `gdn_formats/sglang_46059_7397386194.diff` (sha256 `4e405432...3ac3bb0`). That file is `gh pr diff 46059` at `7397386194`, the 5 files of the PR. The image applies the same file.
- The head of #46059 is `bb1be05646` on 2026-11-18. This branch does not follow it.
- The rotation is on by default. `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0` turns it off.

## AITER

The image uses the AITER of the base image, [`285579b621`](https://github.com/ROCm/aiter/commit/285579b6211b0da30f35b91a890c29c823597cd1), with one edit:

| Edit | Change | Source |
| --- | --- | --- |
| The MXFP4 quant with the Hadamard-32 rotation | The files of [aiter#6734](https://github.com/ROCm/aiter/pull/6734) at `72d86b515f`: `aiter/ops/quant.py`, `csrc/include/quant.h`, `csrc/include/rocm_ops.hpp` and `csrc/kernels/quant_kernels.cu` | `gdn_formats/aiter6734_72d86b515f.diff` (sha256 `e0365856...3722e1d`) |

- The build removes `module_quant.so`. AITER then builds the changed quant module with its JIT at the first use.
- On gfx950, the merged quant kernel of aiter#6734 (`55645da932`) is the same. It only adds a gfx950 check.
- FlyDSL is the FlyDSL of the base image.

## sgl-amd-mono

The plugin comes from [`omirosh/sgl-amd-mono`](https://github.com/omirosh/sgl-amd-mono). The image does not use the pin of `python/pyproject_other.toml`.

| Arm | Plugin commit | Branch | Pull request |
| --- | --- | --- | --- |
| Gate image | [`66ae2fae4b`](https://github.com/omirosh/sgl-amd-mono/commit/66ae2fae4b1070c4a0590619c12eba3ceda3a953) | `q38v3/6-layer-block-measured-66ae2fa` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154), an earlier head |
| A | [`11988b9d7b`](https://github.com/omirosh/sgl-amd-mono/commit/11988b9d7b9c6cd4d2a5c13ffa35fb3a6d953861) | `q38v3/7-hadamard` | [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), draft |
| B | the head of `q38v3/8-layer-block-nofse` | `q38v3/8-layer-block-nofse` | none yet, a draft stacked on [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156) |

- [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156) adds the `qh` and `oh` builds. These builds read the rotated projections of #46059.
- Arm B adds `layer_block` without the fused shared expert. Then a step of 16 rows runs `layer_block` with and without `--enforce-shared-experts-fusion`.
- The drafts can get a rebase. The arm logs and `/etc/sglang-build/manifest.txt` of the image give the plugin commit of each run.

## Image

The job of `qwen38/spec_decode_mi355/mono_plugin/gdn_formats/` builds the images on the node.

1. `gdn_formats/Dockerfile` starts from `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261028@sha256:b51edf074b567263afb19be9108756645ee00d0604374b6e4c5006caccaa17fe` (SGLang `5bb7ecc2c2`, AITER `285579b621`).
2. It applies the #46059 diff to `/sgl-workspace/sglang` and the aiter#6734 diff to `/sgl-workspace/aiter`, and removes `module_quant.so`.
3. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
4. It writes `/etc/sglang-build/manifest.txt`.
5. `gdn_formats/gate.sh` runs the AITER quant check and the #46059 unit test on GPU 0. Then it commits the container as `qwen38-gdnfmt:66ae2fa-7397386`.
6. `gdn_formats/h32_check.sh` and `lb_nofse/nofse_check.sh` make the image of an arm from that gate image: `git checkout` of the plugin commit, `pip install --no-deps -e .`, and the new commit in the manifest.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 to 4 without the SGLang diff.

## Runs

The runs are in `qwen38/spec_decode_mi355/mono_plugin/gdn_formats/PLAN.md` and `qwen38/spec_decode_mi355/mono_plugin/lb_nofse/PLAN.md`.

The servers need `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `--attention-backend aiter`, `--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype bfloat16`, `--cuda-graph-config '{"decode":{"backend":"full","max_bs":16,"bs":[1,2,4,8,16]}}'`, `--tp-size 8` and `--monokernel-backend amd`. The YAMLs have the full command, for example `gdn_formats/h_on_nomtp_r1.yaml`.

- The arms with `fse: 'on'` add `--enforce-shared-experts-fusion`. The lb_nofse A/B arms do not add it. The lb_nofse profiles `prof_a_on` and `prof_b_on` add it.
- `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD` selects the rotation (`1`, the default, or `0`).
