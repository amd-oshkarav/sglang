# Week 45, SGLang v0.5.23 with the monokernel plugin and the fused all-reduce RMSNorm

Base image: `lmsysorg/sglang:v0.5.23-rocm10-mi35x`

Branch `collector_w45_qwen_3.8_v0.5.23_add_rms_norm_mono_plugin` starts at the SGLang tag `v0.5.23` (`297e54fd10`). That tag is on `release/v0.5.23`, cut from SGLang main at `2a7987f900` (27 Oct 2026, 08:42 UTC). The branch adds the 13 commits of [`collector_26_10_qwen_3.8_add_rms_norm_mono_plugin`](https://github.com/amd-oshkarav/sglang/tree/collector_26_10_qwen_3.8_add_rms_norm_mono_plugin) (week 44) on that tag. The week 44 base `e4d4a9a7de` is in `v0.5.23`. The 16 SGLang commits from `e4d4a9a7de` to `v0.5.23` do not touch a file of these 13 commits.

The 13 commits:

- #39069, #40708, #40771 and #43839.
- The FlyDSL INT4 QuickReduce (`8cb39392` and three follow-up commits).
- The sgl-amd-mono pin and the week 44 README.
- The two commits of [`add_rms_norm`](https://github.com/amd-oshkarav/sglang/tree/add_rms_norm). They fuse the FlyDSL INT4 QuickReduce with the residual add and RMSNorm (AITER #6369). The second commit lets a 16384-token prefill chunk use that fusion.

Start the server with `--enable-amd-monokernel` and `--enable-aiter-allreduce-fusion`. The branch does not have #43516. The sgl-amd-mono plugin is the decode monokernel. The plugin runs with the fused shared expert of #41899 and with the MXFP4 GDN projections of #43839.

Pull requests already merged into SGLang main before the `v0.5.23` cut are in this collector through the tag. They were not applied again.

## SGLang

| Pull request | Change | In this collector | In stock v0.5.23 | On SGLang main |
| --- | --- | --- | --- | --- |
| [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP | Yes, added on this branch | No | No, still open |
| [#40708](https://github.com/sgl-project/sglang/pull/40708) | 512-expert softmax MoE through AITER topk_gating | Yes, added on this branch | No | No, still open |
| [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce | Yes, added on this branch | No | No, still open |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quantize the Quark MXFP4 shared expert while loading | Yes, through the tag | Yes | Yes, merged 26 Oct 2026 |
| [#43516](https://github.com/sgl-project/sglang/pull/43516) | Qwen3.8 decode layer as two FlyDSL launches | No, sgl-amd-mono replaces it | No | No, still open |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | Quantize the BF16 GDN projections to MXFP4 online | Yes, the week 44 commit, added on this branch | No | Yes, merged 27 Oct 2026, 22:27 UTC, after the cut |
| [`flydsl_reduce_int4` @ `8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | FlyDSL INT4 QuickReduce for eligible bf16 payloads | Yes, added on this branch | No | No |
| [`add_rms_norm`](https://github.com/amd-oshkarav/sglang/tree/add_rms_norm) | FlyDSL INT4 QuickReduce fused with the residual add and RMSNorm | Yes, added on this branch | No | No |

## sgl-amd-mono

The `monokernels_hip` extra in `python/pyproject_other.toml` installs [`omirosh/sgl-amd-mono` @ `c8b3850aef`](https://github.com/omirosh/sgl-amd-mono/commit/c8b3850aef498814f189bf63edc0f6543c138c94). The `all_hip` and `all_hip_rocm724` extras include it. `docker/rocm.Dockerfile` clones the same fork. Start the server with `--enable-amd-monokernel`.

The plugin spec is `qwen38_a95b_mxfp4_hip`. It serves decode steps of 1 to 8 rows on gfx950 with TP8. With NEXTN, it serves the target-verify steps of 1 and 2 requests. The pinned commit is the top of four stacked branches:

| Branch | Change |
| --- | --- |
| [`q38v2/1-k2-fse`](https://github.com/omirosh/sgl-amd-mono/tree/q38v2/1-k2-fse) | K2 runs the fused shared expert of #41899 (513 slots, top 11) |
| [`q38v2/2-verify`](https://github.com/omirosh/sgl-amd-mono/tree/q38v2/2-verify) | K2 serves the NEXTN target-verify steps |
| [`q38v2/3-k1`](https://github.com/omirosh/sgl-amd-mono/tree/q38v2/3-k1) | K1 runs the GDN front of a decode step as one launch (verify steps keep the SGLang GDN) |
| [`q38v2/4-mxfp4`](https://github.com/omirosh/sgl-amd-mono/tree/q38v2/4-mxfp4) | K1 runs the MXFP4 `in_proj_qkvz` and K2 runs the MXFP4 `out_proj` of #43839 |

## AITER

The collector reinstalls AITER main `9059673612` (26 Oct 2026), as in week 44. The stock v0.5.23 image pins AITER `285579b621` (17 Oct 2026). That pin does not have #6369, so the fused all-reduce RMSNorm needs the reinstall. Every pull request below is on AITER main `9059673612`, so it is in this collector through the checkout.

| Pull request | Change | In stock v0.5.23 | On AITER main |
| --- | --- | --- | --- |
| [#4383](https://github.com/ROCm/aiter/pull/4383) | Gluon MXFP4 and MXFP8 quant on gfx950 | Yes | Yes, merged 5 Oct 2026 |
| [#5334](https://github.com/ROCm/aiter/pull/5334) | E=512 softmax topk_gating prefill dispatch | Yes | Yes, merged 20 Sep 2026 |
| [#5835](https://github.com/ROCm/aiter/pull/5835) | Tuned bf16 GEMM configs for Qwen3.8 on gfx950 | Yes | Yes, merged 26 Sep 2026 |
| [#5998](https://github.com/ROCm/aiter/pull/5998) | Preshuffled AFP4WFP4 GEMM config, N=4608 K=8192 | Yes | Yes, merged 1 Oct 2026 |
| [#6145](https://github.com/ROCm/aiter/pull/6145) | Row-group FP4 paged MQA logits | Yes | Yes, merged 6 Oct 2026 |
| [#6369](https://github.com/ROCm/aiter/pull/6369) | Fuse add-RMSNorm and MXFP4 quant into INT4 QuickReduce | No | Yes, merged 19 Oct 2026 |
| [#6398](https://github.com/ROCm/aiter/pull/6398) | Bind load_hip_runtime to the HIP library already loaded | No | Yes, merged 22 Oct 2026 |

## Image

The image is `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_w45_add_rms_norm_mono_plugin`. It is built from source from this branch, with the same steps as the week 44 image:

1. Clone this branch on top of the base image.
2. Build the SGLang kernels for gfx950.
3. Install `sglang[all_hip]`. This also installs sgl-amd-mono at the pin.
4. Reinstall AITER at `9059673612`.
5. Check the commits, the features and the plugin spec.
