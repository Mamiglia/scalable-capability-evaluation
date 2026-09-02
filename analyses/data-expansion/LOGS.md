# logs/

Judge-side inspect `.eval` logs — each sample is one verdict (which of two
transcripts the judge called more capable). Gitignored; this file is the
inventory. Data-expansion runs use
`{experiment}/{bench}/{judge}/{major}-vs-{minor}/phase-{1,2}/*.eval`
(two-phase resumable runs via `scripts/judge/run_judging.py`); older
experiments have their own shapes.

| path | what |
|---|---|
| `judge-pilot/` | First GPQA pilot: 8 judges (gemma-4-31b/26b, gemini-2.5-flash/3.1-flash-lite/3.5-flash, gpt-4.1-nano, gpt-4o, qwen-2.5-72b) over the July-31 collections + Epoch transcripts. |
| `judge-pilot-incomplete/` | Stray incomplete runs from that pilot. |
| `gpqa-pilot-rd2/` | Pilot round 2 — judge set shifts to gemma-4-31b/26b, gemini-3.1-flash-lite, gpt-4.1-nano, gpt-4o, qwen-2.5-72b, qwen3-32b, qwen3.5-122b; prompts show per-transcript MCQ options (PR #1). Input to the hibayes analysis. |
| `data-expansion/{gpqa,aime2025,aime2026}/` | The multi-eval experiment, cheap-6 judge scope: 45 comparisons per bench, both slot orders, both-correct filtering. 2026-08-08/09. |
| `data-expansion-biggap/gpqa/` | Big-gap probe: 3 weak judges on Epoch transcripts with ECI gaps 24–37. |
| `data-expansion-reasoning-ablation/{aime2025,aime2026}/` | include_reasoning=true ablation — judges see the candidates' reasoning (protocol default hides it, final text only). |
| `capability-spoofing/` | First style-spoof judging: `original/` vs `weak_spoofs_strong/` vs `strong_spoofs_weak/`. |
| `capability-spoofing-curve{,-strong,-instruct}/` | Style-vs-capability gap-curve judging under the three generation conditions: soft style copy / "replicate exactly" / instruction variant (matches `data/capability_spoofing/evals*`). |
| `judge-interp-steered/` | Judging of the steered generations in `data/steered_answers/`. |
| `judge_ft/` | Fine-tuned-judge evaluation: qwen3-0.6b and llama-3.2-3b base-vs-finetuned judges, plus `frontier-answers/` sets they judge. |
