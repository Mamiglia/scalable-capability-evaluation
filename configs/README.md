# configs/

What each file is and which script reads it.

## Judge run configs

Read by `scripts/judge/run_judge_pilot.py --config <file>`. Each holds two
required sections and three optional ones:

| key | required | meaning |
|---|---|---|
| `transcripts` | yes | model shortname → eval log path |
| `comparisons` | yes | `[judge, major, minor]` rows, one per run |
| `truncate_fracs` | no | truncation levels; absent means full transcripts |
| `log_root` | no | where logs go; defaults to `logs/gpqa-pilot-rd2` |
| `judge_model_args` | no | per-judge OpenRouter routing (see Routing below) |

### Round 2 — 2026-08-03, endpoint-pinned

| file | shape | notes |
|---|---|---|
| `gpqa_core_rd2_260803.yaml` | 53 comparisons · 17 pairs (7 self) · 9 judges | gap ladder, self-pair controls, weak judges. The main pilot corpus. |
| `gpqa_trunc_rd2_260803.yaml` | 12 comparisons · 8 pairs · 4 judges · fracs 0.1/0.5/0.9 | length-matching control |
| `gpqa_trunc_weak_rd2_260803.yaml` | 16 comparisons · 5 pairs · 6 judges · fracs 0.1/0.5/0.9 | truncation × weak judges |
| `gpqa_trunc_self_rd2_260803.yaml` | 14 comparisons · 7 self-pairs · 3 judges · fracs 0.1/0.5/0.9 | self-pairs under truncation — the control the other truncation configs lack. Written but never run. |
| `gpqa_smoke_rd2_260803.yaml` | 6 comparisons · 3 pairs · 2 judges | end-to-end pipeline check, ~240 calls |

### Round 3 — 2026-08-04, expanded

| file | shape | notes |
|---|---|---|
| `gpqa_expanded_rd3_260804.yaml` | 360 comparisons · 30 pairs · 12 judges · `log_root: logs/gpqa-expanded-260804` | **stale.** Built for the uniform-crossing design; `docs/2026-08-04-expanded-judging-design.md` was since revised to a B1/B2/B3 block structure with 50 pairs and 16 judges. Regenerate before use. Never run. |

### Round 1 — superseded, kept for provenance

Unpinned routing, so their logs are not poolable with round 2+.

| file | shape |
|---|---|
| `judge_pilot.yaml` | 103 comparisons · 23 pairs · 10 judges. Still the default `--config` for `run_judge_pilot.py` and `digest_evals.py`. |
| `judge_trunc_arma.yaml` | 62 comparisons · 11 pairs · 10 judges · fracs 0.1/0.5/0.9 |
| `judge_priority.yaml` | 5 comparisons · 2 pairs · 4 judges — weak judges on big-gap anchors |

`judge_trunc.yaml` and `judge_trunc_weak.yaml` were renamed into the round-2
set; git history has them under the old names.

## Routing

`judge_model_args` maps a judge slug to args passed through to
`eval_set(model_args=...)`, which reach OpenRouter as `extra_body.provider`.

Open-weight judges are pinned to a specific endpoint tag with
`allow_fallbacks: false`. Without a pin, OpenRouter load-balances a slug across
endpoints that differ in quantization, context window and max output — several
of these models have fp4 routes. Uniform fp8 rather than best-available
precision: `qwen3-32b` and `qwen3.6-35b` have no bf16 endpoint anywhere, and
best-available would hand bf16 to the strongest judges, making precision
co-vary with judge capability in exactly the comparison the study rests on.

Proprietary judges (Gemini, GPT, Claude) are effectively single-served and get
the policy flags only.

`provenance/openrouter_endpoints_260804.json` is the endpoint record as fetched
on 2026-08-04, with the pins chosen from it. Providers publish launches and
retirements but not changes to an endpoint that keeps its name, so diffing a
stored snapshot against a fresh fetch is the only way to notice a pinned
endpoint started serving something else. Re-fetch and diff before a run that
matters.

## Other

| path | read by | what |
|---|---|---|
| `dashboard_datasets.yaml` | `scripts/analyses/export_dashboard.py` | datasets in the viewer's dropdown, in menu order |
| `bench/*.yml` | `uv run collect -b` | benchmark task configs (gpqa, gsm8k, mbpp, core-bench easy/medium/hard) |
| `models/*.yml` | `uv run collect -m` | model configs for transcript collection |
| `experiments/**/*.yml` | `uv run judge --config_path` | `ExperimentConfig` runs — the older single-experiment path, separate from the `run_judge_pilot.py` config format above |
| `irt/gpqa.yaml` | `uv run irt` | IRT fitting over the Epoch response matrix |

## Naming

Round 2 onward: `gpqa_<set>_rd<n>_<yymmdd>.yaml`. Round-1 files keep their
original `judge_*` names.
