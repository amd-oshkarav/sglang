# Week 44, monokernel plugin

Base image: `lmsysorg/sglang:v0.5.22-rocm10-mi35x`

Branch `collector_26_10_qwen_3.8_mono_plugin`, cut from SGLang main at `e4d4a9a7de` (27 Oct 2026). AITER is `ROCm/aiter` main at `9059673612`.

This branch is [`collector_26_10_qwen_3.8`](https://github.com/amd-oshkarav/sglang/tree/collector_26_10_qwen_3.8) (`063fdea898`) with a different decode monokernel. The two branches start from the same main commit. They have the same commits for #39069, #40708, #40771, #43839 and the FlyDSL INT4 QuickReduce. This branch does not have #43516. The sgl-amd-mono plugin is the decode monokernel. The plugin runs with the fused shared expert of #41899 and with the MXFP4 GDN projections of #43839.

Pull requests already merged into SGLang main are in this collector because the branch starts from that main. They were not applied again.

## SGLang

| Pull request | Change | In this collector | In stock v0.5.22 | On SGLang main |
| --- | --- | --- | --- | --- |
| [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP | Yes, added on this branch | No | No, still open |
| [#40708](https://github.com/sgl-project/sglang/pull/40708) | 512-expert softmax MoE through AITER topk_gating | Yes, added on this branch | No | No, still open |
| [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce | Yes, added on this branch | No | No, still open |
| [#41899](https://github.com/sgl-project/sglang/pull/41899) | Quantize the Quark MXFP4 shared expert while loading | Yes, through the main base | No | Yes, merged 26 Oct 2026 |
| [#43516](https://github.com/sgl-project/sglang/pull/43516) | Qwen3.8 decode layer as two FlyDSL launches | No, sgl-amd-mono replaces it | No | No, still open |
| [#43839](https://github.com/sgl-project/sglang/pull/43839) | Quantize the BF16 GDN projections to MXFP4 online | Yes, added on this branch | No | No, still open |
| [`flydsl_reduce_int4` @ `8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | FlyDSL INT4 QuickReduce for eligible bf16 payloads | Yes, added on this branch | No | No |

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

The collector reinstalls AITER main `9059673612`. The stock v0.5.22 image pins AITER `e7d2453` (25 Sep 2026). Every pull request below is on that AITER main, so it is in this collector through the checkout.

| Pull request | Change | In stock v0.5.22 | On AITER main |
| --- | --- | --- | --- |
| [#4383](https://github.com/ROCm/aiter/pull/4383) | Gluon MXFP4 and MXFP8 quant on gfx950 | No | Yes, merged 5 Oct 2026 |
| [#5334](https://github.com/ROCm/aiter/pull/5334) | E=512 softmax topk_gating prefill dispatch | Yes | Yes, merged 20 Sep 2026 |
| [#5835](https://github.com/ROCm/aiter/pull/5835) | Tuned bf16 GEMM configs for Qwen3.8 on gfx950 | No | Yes, merged 26 Sep 2026 |
| [#5998](https://github.com/ROCm/aiter/pull/5998) | Preshuffled AFP4WFP4 GEMM config, N=4608 K=8192 | No | Yes, merged 1 Oct 2026 |
| [#6145](https://github.com/ROCm/aiter/pull/6145) | Row-group FP4 paged MQA logits | No | Yes, merged 6 Oct 2026 |
| [#6398](https://github.com/ROCm/aiter/pull/6398) | Bind load_hip_runtime to the HIP library already loaded | No | Yes, merged 22 Oct 2026 |

## Image

The image is `amdsiloai/sglang-private:v0.5.22-rocm10-mi35x_collector_26_qr_flydsl_int4_mono_plugin`. It is built from source from this branch, with the same steps as the `collector_26_10_qwen_3.8` image:

1. Clone this branch on top of the base image.
2. Build the SGLang kernels for gfx950.
3. Install `sglang[all_hip]`. This also installs sgl-amd-mono at the pin.
4. Reinstall AITER at `9059673612`.
5. Check the commits, the features and the plugin spec.
