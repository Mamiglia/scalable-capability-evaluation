# data/

Collected benchmark transcripts (inspect `.eval` logs) — the candidate answers
that judging runs compare. Gitignored except this file, `collection_runs.md`,
and the endpoint snapshot. Collection layout:
`{bench}/{task_args_hash}/{model}/{repeat}/*.eval`. How each collection was run
and verified: `collection_runs.md`.

| path | what |
|---|---|
| `gpqa/2026-07-31T*.eval` | July-31 pilot GPQA-diamond collections (gemma-4-31b/26b, gemini-2.5-flash, gemini-3.1-flash-lite, qwen3.5-122b, qwen3.6-35b, qwen3-32b), flat files. Judged in `logs/gpqa-pilot-rd2`; path map in `configs/judge_pilot.yaml`. |
| `gpqa/63793eb6/` | Data-expansion GPQA-diamond collection (cot=true, epochs=4), 2026-08-08 seedless run: the cheap 6 + gemini-3.1-flash-lite + gpt-4.1, o4-mini, claude-sonnet-4.5. `gemini-3.1-flash-lite-rescored/` = tolerant-ANSWER rescore of the gemini log (`scripts/judge/rescore_mcq.py`); judging configs point at the rescored copy. |
| `aime2025/44136fa3/`, `aime2026/44136fa3/` | Same model grid on AIME 2025 / 2026 (epochs=4, free-form integer answers). |
| `capability_spoofing/` | Style-vs-capability answer sets converted to eval logs, one per (model, condition): `evals/` soft style copy, `evals_strong/` "replicate exactly" copy, `evals_instruct/` confident/uncertain instruction variant. |
| `steered_answers/` | Judge-interp steered generations as eval logs: `unsteered/`, `random_a_p0/`, `random_a_p0.05/`, `random_a_p1/`. |
| `rewrite_*.jsonl` | Judge-interp rewrite sets — transcripts rewritten into another style, one JSON row per rewrite (model, item, target, answer, rewriter): `pilot` 20, `style3` 120, `style3_full` 512, `local` 4943 rows. |
| `openrouter_endpoints_2026-08-08.json` | OpenRouter endpoint snapshot at data-expansion collection time — which endpoints/quantizations were live per model. |
| `collection_runs.md` | Run log: what was collected and judged when, verification, incidents (seed abort, gemini rescore, cap trip). |
