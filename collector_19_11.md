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
