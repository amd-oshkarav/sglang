# collector_19_11

Branch `collector_19_11` starts from SGLang main [`b1970bc473`](https://github.com/sgl-project/sglang/commit/b1970bc473cfff1731c068e7d23e9e61b794b0c0) (`b1970bc473cfff1731c068e7d23e9e61b794b0c0`), 2026-11-19 00:52 -0800:

`[Feat] cli: Export Cookbook commands for every config-driven LLM/VLM page (#50106)`

Each row below is one commit on top of that main. The source list is the SGLang table on [`collector_09_11_qwen_3.8_stack_p2`](https://github.com/amd-oshkarav/sglang/tree/collector_09_11_qwen_3.8_stack_p2). Merge status was checked against `sgl-project/sglang` main, not against that table.

## Ported features

| Commit | Pull request | Feature |
| --- | --- | --- |
| `7502002d4c` | [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP |
| `cea365f745` | [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce |
| `d1b489d47f` | [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | Use FlyDSL INT4 QuickReduce for contiguous bf16 |
| `d10534a597` | [#47219](https://github.com/sgl-project/sglang/pull/47219) | Fuse add-RMSNorm into FlyDSL INT4 QuickReduce |
| `af1219fd5e` | [#47433](https://github.com/sgl-project/sglang/pull/47433) | Use one ReplaySSM pipeline stage on gfx950 |
| `bc161b2bb6` | [#45529](https://github.com/sgl-project/sglang/pull/45529) | Fuse self-attn prepare on gfx950 (norm, RoPE, gate) |
| `d9313039d0` | [#39595](https://github.com/sgl-project/sglang/pull/39595) | Add the FlyDSL GDN prefill backend |

#40771 and #47219 were ported as those features. Their open pull requests also contain other files, which were not brought over.

## Model and sweep

Qwen3.8-2.4T TP8 FP4 on MI355X. This branch is SGLang `f3371c01c2` (`collector_19_11`). The image that serves it is the collector image plus one layer that installs [sgl-amd-mono#154](https://github.com/sgl-project/sgl-amd-mono/pull/154).

AITER is `90596736129ccc0bcf720686cf202067a11f110d` (2026-10-26). That commit is before [aiter#5988](https://github.com/ROCm/aiter/pull/5988) and does not contain [aiter#7441](https://github.com/ROCm/aiter/pull/7441). The image adds that pull request's two GEMM tables from `ec91f61f0c550811cf121f0b0dc667c09524b161` and does not apply the P-1b edit.

The monokernel is [sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) `main` at `32a122031b7b63858c1610370b49a3bf9a215997`, plus the five commits of PR 154 through `05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68`.

## Benchmark

The sweep is `docs/examples/tonight/qwen_3.8_collector_19_11_mono.yaml` in `AMD-AGI/inference-testing`. It starts the mono image below and runs `sglang_bench_serve` on random data (`random-range-ratio` 0.8, `request-rate` inf). All twelve cells run:

| ISL | OSL | Conc | Prompts |
| --- | --- | --- | --- |
| 8192 | 1024 | 4, 8, 16, 32, 64, 128, 256 | 32, 64, 96, 128, 256, 384, 768 |
| 60000 | 600 | 4, 16, 64 | 32, 96, 256 |
| 128000 | 1024 | 4, 16 | 32, 96 |

GPQA follows the sweep (`sglang_run_eval`, 32 threads, `max-tokens` 100000, temperature 1.0, top-p 0.95, thinking enabled). RULER `niah_single_2` is in the file and commented out. The finished run is [8ddc5f7ab5f545ee8ea7df576191e8d4](https://adb-4032599967441609.9.azuredatabricks.net/ml/experiments/1169297213372844/runs/8ddc5f7ab5f545ee8ea7df576191e8d4?o=4032599967441609).

The server uses `AITER_FLYDSL_STAGE2_FP8=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=512`, `--enable-aiter-allreduce-fusion`, `--enforce-shared-experts-fusion`, `--linear-attn-prefill-backend flydsl` and `--enable-amd-monokernel`. Page size is 16, unified attention and GDN online MXFP4 are on, and the prefill chunk is 16384 tokens.

## Images

The serving image is `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736_mono_32a12203_pr154`. It is the base image plus the sgl-amd-mono layer.

### sgl-amd-mono#154 layer

Dockerfile: `docker/collector_19_11_mono/Dockerfile` in `AMD-AGI/inference-testing`. The sweep config that builds this tag is `docs/examples/tonight/qwen_3.8_collector_19_11_mono.yaml`, and its `dockerfile.path` is `/app/repos/inference-testing/docker/collector_19_11_mono/Dockerfile`.

This file starts `FROM` `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736` and leaves SGLang and AITER as that image built them. It installs the monokernel plugin:

1. Uninstall any `sgl-amd-mono` already present and clone [sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) `main`.
2. Detach at `32a122031b7b63858c1610370b49a3bf9a215997` and refuse any other commit.
3. Check that SGLang's `register()` already accepts `model_type`. PR 154 calls it that way.
4. Fetch `pull/154/head` and cherry-pick these five commits, in order, onto that main pin. The pull request is open, and its base is one commit behind main, so the image records the main pin and the cherry-picks rather than a single remote SHA.
   - `7fc7b73e6f40e52fd6a9eada64508b148d5ba71c` runs the tail of a 16-row step as one `layer_block` launch.
   - `289c1a2d611e085db6a2b83859b1f4db614e4e9f`, `1ec6f748a8251d572fafac22e38e257684b271d5`, `34d8f8cbb88050788e62e880601cca08838ef5ac` and `05b9dd9fabf8418cfe8efa7c2d9ab8780be2fb68` are the tests for that launch. `05b9dd9f` is the head.
5. Require that main is an ancestor and that exactly those five commits sit on top of it.
6. `pip install --no-deps -e .` and require the `sgl_amd_mono` entry point under `sglang.srt.plugins`.

`layer_block` runs when the server is started with `--enable-amd-monokernel` and `--enforce-shared-experts-fusion`.

### Base image

The tag is `amdsiloai/sglang-private:v0.5.23-rocm10-mi35x_collector_19_11_f3371c01_aiter_90596736`.

Dockerfile: `docker/collector_19_11/Dockerfile` in `AMD-AGI/inference-testing`. The config that builds this tag is `docs/examples/collector/qwen_3.8_collector_19_11.yaml`, and its `dockerfile.path` is `/app/repos/inference-testing/docker/collector_19_11/Dockerfile`.

It starts from `lmsysorg/sglang:v0.5.23-rocm10-mi35x` and builds for gfx950:

1. Install Rust if the base image has no `cargo`.
2. Clone [amd-oshkarav/sglang](https://github.com/amd-oshkarav/sglang) branch `collector_19_11` and detach at `f3371c01c2f9292a0f4e44ed2de212959c0342f6`, with submodules. The checkout is used as it is. On top of SGLang main `b1970bc473` that commit carries #39069 (bundled MTP draft path), #40771 (QuickReduce before custom all-reduce), FlyDSL INT4 QuickReduce (`8cb39392`), #47219 (add-RMSNorm in that QuickReduce), #47433 (one ReplaySSM stage on gfx950), #45529 (fused self-attention prepare) and #39595 (FlyDSL GDN prefill).
3. Build the ROCm kernels from `sgl-kernel` or `python/sglang/kernels/aot`.
4. Install that checkout in editable mode. The extra is `all_hip` when the base torch is 2.9 or 2.10, and `all_hip_rocm724` otherwise. Torch and Triton stay pinned to the base image, `sgl-eval==0.1.2` is installed, and the build stops if a CUDA wheel was pulled in.
5. Replace AITER. Clone [ROCm/aiter](https://github.com/ROCm/aiter) `main` and detach at `90596736129ccc0bcf720686cf202067a11f110d` (2026-10-26). That commit is 228 commits before `8f4fe9e5` and is before [aiter#5988](https://github.com/ROCm/aiter/pull/5988). It does not contain [aiter#7441](https://github.com/ROCm/aiter/pull/7441). Apply `aiter_7441.patch`, which is that pull request (`ec91f61f0c`) and adds the two Qwen3.8 TP8 a4w4 GEMM tables (`qwen38_2p4t_a4w4_blockscale_tuned_gemm.csv` and the untuned table, N 4608, K 8192). Reinstall AITER from that tree and import `QuickAllReduceInt4`. The P-1b one-line edit is not applied.
6. Check again that SGLang is still `f3371c01` and that the FlyDSL GDN prefill backend is present.

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
