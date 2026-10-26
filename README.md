# Week 43

Base image: `lmsysorg/sglang:v0.5.21-rocm10-mi35x`

Branch `collector_24_10_qwen3.8_cafirst_noflydslint4`. This list is what the collector uses. It does not say which of these were already merged.

- SGLang `main` @ `e1c4b273af`
- [#39069](https://github.com/sgl-project/sglang/pull/39069) Infer the draft path for bundled MTP
- [#40708](https://github.com/sgl-project/sglang/pull/40708) 512-expert softmax MoE through AITER topk_gating
- [#43516](https://github.com/sgl-project/sglang/pull/43516) Qwen3.8 decode layer as two FlyDSL launches
- [#43839](https://github.com/sgl-project/sglang/pull/43839) Quantize the BF16 GDN projections to MXFP4 online
- AITER `main` @ `9059673612`
