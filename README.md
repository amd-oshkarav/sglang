# Week 47 gdn_formats and lb_nofse, monokernel plugin

Image: `qwen38-gdnfmt:<sha7>-7397386`. The image is local only. Each job builds it on the node and does not push it to a registry. `<sha7>` is the sgl-amd-mono plugin commit of the arm.

This branch is SGLang `5bb7ecc2c2` (2026-10-28), the SGLang of the base image, plus one commit for [#46059](https://github.com/sgl-project/sglang/pull/46059). The image code is in `qwen38/spec_decode_mi355/mono_plugin/gdn_formats/` of [AMD-AGI/amd-silo-ai-tiger-teams-sglang](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/tree/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats) (commit `5540c7dc` of branch `olga`). The section "Image" gives the commit of each build.

The image has an AITER edit too. AITER is not in this repo. The section "AITER" gives the edit.

## SGLang

| Pull request | Change | On this branch | On SGLang main (2026-11-18) |
| --- | --- | --- | --- |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quark MXFP4 fused shared expert | Yes, through the base `5bb7ecc2c2` | Yes, merged 2026-10-26 |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | MXFP4 GDN projections, online | Yes, through the base `5bb7ecc2c2` | Yes, merged 2026-10-27 |
| [#46059](https://github.com/sgl-project/sglang/pull/46059) | Hadamard-32 rotation of the online MXFP4 GDN projections | `eb7811e22c`, the PR at `7397386194` | No, open |

- `eb7811e22c` applies [`gdn_formats/sglang_46059_7397386194.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/sglang_46059_7397386194.diff) (sha256 `4e405432...3ac3bb0`). That file is `gh pr diff 46059` at `7397386194`, the 5 files of the PR. The image applies the same file.
- The head of #46059 is `bb1be05646` on 2026-11-18. This branch does not follow it.
- The rotation is on by default. `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD=0` turns it off.

## AITER

The image uses the AITER of the base image, [`285579b621`](https://github.com/ROCm/aiter/commit/285579b6211b0da30f35b91a890c29c823597cd1), with one edit:

| Edit | Change | Source |
| --- | --- | --- |
| The MXFP4 quant with the Hadamard-32 rotation | The files of [aiter#6734](https://github.com/ROCm/aiter/pull/6734) at `72d86b515f`: `aiter/ops/quant.py`, `csrc/include/quant.h`, `csrc/include/rocm_ops.hpp` and `csrc/kernels/quant_kernels.cu` | [`gdn_formats/aiter6734_72d86b515f.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/aiter6734_72d86b515f.diff) (sha256 `e0365856...3722e1d`) |

- The build removes `module_quant.so`. AITER then builds the changed quant module with its JIT at the first use.
- On gfx950, the merged quant kernel of aiter#6734 (`55645da932`) is the same. It only adds a gfx950 check.
- FlyDSL is the FlyDSL of the base image.

## sgl-amd-mono

The plugin comes from [`omirosh/sgl-amd-mono`](https://github.com/omirosh/sgl-amd-mono). The image does not use the pin of `python/pyproject_other.toml`.

| Arm | Plugin commit | Branch | Pull request |
| --- | --- | --- | --- |
| Gate image | [`66ae2fae4b`](https://github.com/omirosh/sgl-amd-mono/commit/66ae2fae4b1070c4a0590619c12eba3ceda3a953) | `q38v3/6-layer-block-measured-66ae2fa` | [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154), an earlier head |
| `h_on` of gdn_formats | [`db4e594099`](https://github.com/omirosh/sgl-amd-mono/commit/db4e594099a2cc3945de9b9b57aff1d7b23d1de1) | `q38v3/7-hadamard-measured-db4e594` | [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), the branch before a rebase |
| A | [`11988b9d7b`](https://github.com/omirosh/sgl-amd-mono/commit/11988b9d7b9c6cd4d2a5c13ffa35fb3a6d953861) | `q38v3/7-hadamard-measured-11988b9` | [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156), draft |
| B | [`dc77ad6094`](https://github.com/omirosh/sgl-amd-mono/commit/dc77ad6094c196439c549bc151cf9a4c4a5dde1d) | `q38v3/8-layer-block-nofse-measured-dc77ad6` | [sgl-amd-mono#159](https://github.com/sgl-project/sgl-amd-mono/pull/159), draft, stacked on [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156) |

- [sgl-amd-mono#156](https://github.com/sgl-project/sgl-amd-mono/pull/156) adds the `qh` and `oh` builds. These builds read the rotated projections of #46059.
- Arm B adds `layer_block` without the fused shared expert. Then a step of 16 rows runs `layer_block` with and without `--enforce-shared-experts-fusion`.
- Both drafts got a rebase after the runs. The `-measured-` branches of the fork keep the commits of the runs. The arm logs and `/etc/sglang-build/manifest.txt` of the image give the plugin commit of each run.

## Image

Two Crusoe jobs built the images on the node. Each job read the build files from the login clone of `AMD-AGI/amd-silo-ai-tiger-teams-sglang`. The links in the steps give each file at the HEAD of that clone at the build time.

| Job and node | Image | Built (UTC) | Script | Commit |
| --- | --- | --- | --- | --- |
| 292173, `crsuse2-m2m-v3-257` | `qwen38-gdnfmt:66ae2fa-7397386`, the gate image | 2026-11-16 23:29 | `gdn_formats/gate.sh` | [`645b7341`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/645b73416223f75f983ed3d2f5ed1d3ad5686b9b) |
| 292173, `crsuse2-m2m-v3-257` | `qwen38-gdnfmt:db4e594-7397386`, arm `h_on` | 2026-11-17 09:14 | `gdn_formats/h32_check.sh` | [`9e734fb8`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/9e734fb80d2bf767b3a63d9fcd0ceea3b9cc8cf8) |
| 297398, `crsuse2-m2m-v3-107` | `qwen38-gdnfmt:66ae2fa-7397386`, the gate image | 2026-11-18 23:13 | `gdn_formats/gate.sh` | [`5540c7dc`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/5540c7dcb929e7012e5e83dbd9b13c2830aa3952) |
| 297398, `crsuse2-m2m-v3-107` | `qwen38-gdnfmt:dc77ad6-7397386`, arm B | 2026-11-18 23:42 | `lb_nofse/nofse_check.sh` | [`5540c7dc`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/5540c7dcb929e7012e5e83dbd9b13c2830aa3952) |
| 297398, `crsuse2-m2m-v3-107` | `qwen38-gdnfmt:11988b9-7397386`, arm A | 2026-11-19 00:03 | `lb_nofse/nofse_check.sh` | [`5540c7dc`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/commit/5540c7dcb929e7012e5e83dbd9b13c2830aa3952) |

- The Dockerfile, the two diffs and `gate.sh` have one commit, `a4963b38`. Thus both gate images come from the same files. The links give these files at `5540c7dc`.
- `h32_check.sh` changed in `f28152ab`, after the build of the `h_on` image. The change is in the tests only. The link gives the file at `9e734fb8`.
- ITT did not build an image. The scripts give each image the label of its build arguments. ITT found that label and ran the image.

1. [`gdn_formats/Dockerfile`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/Dockerfile) starts from `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261028@sha256:b51edf074b567263afb19be9108756645ee00d0604374b6e4c5006caccaa17fe` (SGLang `5bb7ecc2c2`, AITER `285579b621`).
2. It applies [`gdn_formats/sglang_46059_7397386194.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/sglang_46059_7397386194.diff) to `/sgl-workspace/sglang` and [`gdn_formats/aiter6734_72d86b515f.diff`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/aiter6734_72d86b515f.diff) to `/sgl-workspace/aiter`, and removes `module_quant.so`.
3. It checks out the plugin at `PLUGIN_SHA` in `/sgl-workspace/sgl-amd-mono` and installs it with `pip install --no-deps -e .`.
4. It writes `/etc/sglang-build/manifest.txt`.
5. [`gdn_formats/gate.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/gate.sh) runs the AITER quant check and the [#46059](https://github.com/sgl-project/sglang/pull/46059) unit test on GPU 0. Then it commits the container as `qwen38-gdnfmt:66ae2fa-7397386`.
6. [`gdn_formats/h32_check.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/9e734fb80d2bf767b3a63d9fcd0ceea3b9cc8cf8/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/h32_check.sh) and [`lb_nofse/nofse_check.sh`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/5540c7dcb929e7012e5e83dbd9b13c2830aa3952/qwen38/spec_decode_mi355/mono_plugin/lb_nofse/nofse_check.sh) make the image of an arm from that gate image: `git checkout` of the plugin commit, `pip install --no-deps -e .`, and the new commit in the manifest.

For a build from source, check out this branch in `/sgl-workspace/sglang` of the base image. Then do steps 2 to 4 without the SGLang diff.

## Runs

The runs are in [`qwen38/spec_decode_mi355/mono_plugin/gdn_formats/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/PLAN.md) and [`qwen38/spec_decode_mi355/mono_plugin/lb_nofse/PLAN.md`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/mono_plugin/lb_nofse/PLAN.md).

The servers need `SGLANG_USE_AITER=1`, `SGLANG_USE_AITER_UNIFIED_ATTN=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `--attention-backend aiter`, `--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype bfloat16`, `--cuda-graph-config '{"decode":{"backend":"full","max_bs":16,"bs":[1,2,4,8,16]}}'`, `--tp-size 8` and `--monokernel-backend amd`. The YAMLs have the full command, for example [`gdn_formats/h_on_nomtp_r1.yaml`](https://github.com/AMD-AGI/amd-silo-ai-tiger-teams-sglang/blob/891d4bf32839b3d365e24cf6295040c08102901c/qwen38/spec_decode_mi355/mono_plugin/gdn_formats/h_on_nomtp_r1.yaml).

- The arms with `fse: 'on'` add `--enforce-shared-experts-fusion`. The lb_nofse A/B arms do not add it. The lb_nofse profiles `prof_a_on` and `prof_b_on` add it.
- `SGLANG_ROCM_GDN_ONLINE_MXFP4_HADAMARD` selects the rotation (`1`, the default, or `0`).
