# collector_23_11

Branch `collector_23_11` starts from SGLang main [`ac0380c7fa`](https://github.com/sgl-project/sglang/commit/ac0380c7faf9cece3ca32ffe75815af7c585acea) (`ac0380c7faf9cece3ca32ffe75815af7c585acea`), 2026-11-23 21:44 +0800:

`[Perf] [AMD] Qwen3.8: Serve QSA decode with AITER's fused FlyDSL sparse decode on gfx950 (#45196)`

Each row below is one commit on top of that main, cherry-picked from [`collector_19_11`](https://github.com/amd-oshkarav/sglang/tree/collector_19_11) in that branch's order. Merge status was checked against this main: none of the pull requests below is merged, and main has no other commit that carries the same change.

## Ported features

| Commit | From `collector_19_11` | Pull request | Feature |
| --- | --- | --- | --- |
| `ac3c4028d5` | `7502002d4c` | [#39069](https://github.com/sgl-project/sglang/pull/39069) | Infer the draft path for bundled MTP |
| `90b2a38049` | `cea365f745` | [#40771](https://github.com/sgl-project/sglang/pull/40771) | Prefer QuickReduce before custom all-reduce |
| `4c815a8602` | `d1b489d47f` | [`8cb39392`](https://github.com/amd-oshkarav/sglang/commit/8cb39392b6dbcce24dbfae6d6a3315c2f98aa6e7) | Use FlyDSL INT4 QuickReduce for contiguous bf16 |
| `cff152e076` | `d10534a597` | [#47219](https://github.com/sgl-project/sglang/pull/47219) | Fuse add-RMSNorm into FlyDSL INT4 QuickReduce |
| `391caa0094` | `af1219fd5e` | [#47433](https://github.com/sgl-project/sglang/pull/47433) | Use one ReplaySSM pipeline stage on gfx950 |
| `db8eeb2708` | `bc161b2bb6` | [#45529](https://github.com/sgl-project/sglang/pull/45529) | Fuse self-attn prepare on gfx950 (norm, RoPE, gate) |
| `c69ef0ed88` | `d9313039d0` | [#39595](https://github.com/sgl-project/sglang/pull/39595) | Add the FlyDSL GDN prefill backend |
| `a0406b75eb` | `6437d57827` | [#50069](https://github.com/sgl-project/sglang/pull/50069) | Pick the all-reduce kernel from a tune table |
| `6e134df6b6` | `cf01f8effa` | [#50069](https://github.com/sgl-project/sglang/pull/50069) | All-reduce keyword fix for the tuned selection |

#40771 and #47219 were ported as those features. Their open pull requests also contain other files, which were not brought over.

`4c815a8602` had one conflict in `quick_all_reduce.py`. Main [#50803](https://github.com/sgl-project/sglang/pull/50803) removed the unreachable `qr_max_size < 1` warning and reworded the comment after it. The port keeps that removal and the FlyDSL workspace comment. Every other ported file matches `collector_19_11` apart from main's own changes since the two branches split.
