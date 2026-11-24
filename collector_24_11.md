# collector_24_11

Branch `collector_24_11` starts from SGLang main [`88dc7501f5`](https://github.com/sgl-project/sglang/commit/88dc7501f58f026a18357b8ad4ef1f280e1fc2b4) (`88dc7501f58f026a18357b8ad4ef1f280e1fc2b4`), 2026-11-24 00:56 -0800:

`[CI] pr-states: Drop the duplicate dispatch from pr-test and pr-test-extra; workflow_run already refreshes on reruns (#52788)`

Each row below is one commit on top of that main, cherry-picked from [`collector_23_11`](https://github.com/amd-oshkarav/sglang/tree/collector_23_11) in that branch's order. Merge status was checked against this main (the tip of `sgl-project/sglang` main when the branch was made): none of the pull requests below is merged, and every commit applied without a conflict.

## Ported features

| Commit | From `collector_23_11` | Pull request | Feature |
| --- | --- | --- | --- |
| `a756297335` | `ac3c4028d5` | [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP |
| `665ddf2c74` | `90b2a38049` | [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce |
| `3e49b12043` | `4c815a8602` | [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | Use FlyDSL INT4 QuickReduce for contiguous bf16 |
| `5ef3f38228` | `cff152e076` | [#47219](https://github.com/sgl-project/sglang/pull/47219) | Fuse add-RMSNorm into FlyDSL INT4 QuickReduce |
| `7e8d8d53ae` | `391caa0094` | [#47433](https://github.com/sgl-project/sglang/pull/47433) | Use one ReplaySSM pipeline stage on gfx950 |
| `4f1547fd70` | `db8eeb2708` | [#45529](https://github.com/sgl-project/sglang/pull/45529) | Fuse self-attn prepare on gfx950 (norm, RoPE, gate) |
| `4af7130f0d` | `c69ef0ed88` | [#39595](https://github.com/sgl-project/sglang/pull/39595) | Add the FlyDSL GDN prefill backend |

The diff of each commit is identical to its `collector_23_11` original (same `git patch-id`).

#40771 and #47219 were ported as those features. Their open pull requests also contain other files, which were not brought over.

## Not ported

| `collector_23_11` commit | Pull request | Feature | Why |
| --- | --- | --- | --- |
| `a0406b75eb` | [#50069](https://github.com/sgl-project/sglang/pull/50069) | Pick the all-reduce kernel from a tune table | The tuned all-reduce injection is left out on purpose. |
| `6e134df6b6` | [#50069](https://github.com/sgl-project/sglang/pull/50069) | All-reduce keyword fix for the tuned selection | It only repairs the injection. Without the injection, `all_reduce` and `_resolve_outplace_all_reduce_method` already agree on `should_use_quick_allreduce`. |

Because the injection is not here, `SGLANG_TUNED_EAGER_CONFIG` and `SGLANG_TUNED_GRAPH_CONFIG` do nothing on this branch.
