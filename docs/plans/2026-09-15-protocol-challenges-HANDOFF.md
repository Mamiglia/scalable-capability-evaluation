# Protocol challenges — handoff

Status checked 2026-09-15 against the local implementation, published judging records through `LASR-G5/judging@08dfef81` (hidden-reasoning sweep), and the project's benchmark pin. This supersedes collection-status statements in the [earlier challenge plan](2026-09-14-protocol-challenges.md) and older launch handoffs. No collection was launched while writing this document.

## Scope and decisions already made

The core experiment predicts future models' domain-specific HARD-item capability, θ*, from correct first-attempt transcripts on selected EASY items. Future generalisation and difficulty generalisation (original challenges **1 and 2**) are already built into the project. They are not new challenge campaigns.

The primary metric is MSE(θ̂, θ*), accompanied by valid-target coverage. Transcript transformations keep the underlying subject rollouts, EASY/HARD split and θ* fixed. Apply the same viewing condition to calibration and target inputs. Reuse identical inputs and existing judgments after verifying their provenance; do not rerun every judge/domain/semester by default.

The completed direct campaigns used previous-semester references only. Current generic supervised code supports references plus historical models with task-specific labels; pairwise still uses references only. Preserve the completed campaign's calibration cohort in a matched comparison. Expanding history would be a separate condition.

| Challenge | Current position | Remaining work |
| --- | --- | --- |
| 1. Future generalisation | Existing core | Use the existing semester tasks. |
| 2. Difficulty generalisation | Existing core | Keep the frozen EASY evidence / HARD θ* setup. |
| 3. Truncation / optional paraphrasing | Cyber and coding rollouts collected; full-transcript direct judging published, with a coding freeze mismatch described below | Select a small condition set; collect transformed-input judgments and run matched predictions. |
| 4. Hidden reasoning | Pairwise off done; direct off collected and published for every primary maths/science judge whose inputs change (17 conditions, 2025-I onward); paired visible/hidden comparison produced | Finish and evaluate the remaining baselines. Supplemental maths judges (Llama-3.3-70B, WizardLM) are uncovered. |
| 5. Reasoning introduction | One historical maths case; pairwise essentially done and direct forecasts available | Analyse existing forecasts. |
| 6. Style robustness | Dropped as a standalone challenge | Optional paraphrasing belongs under challenge 3; no distilled-model campaign planned. |
| 7. Domain generalisation | An analysis/refitting task where coverage overlaps | Match source evidence to destination labels, recalibrate/refit, and evaluate. |

## 3. Truncation and optional paraphrasing

Restrict this challenge to **coding and cyber**. Short-task-to-long-task prediction is outside the agreed scope. Deterministic prefix truncation already exists through `JudgeConfig.max_length`: the limit is in **characters**, applied to the rendered transcript. Use the same limit on calibration and target transcripts. Limits, subjects and judges remain to be selected; no truncated or paraphrased challenge condition is recorded as collected here.

Subject collection is no longer the blanket blocker described in the original plan:

- **Cyber:** InterCode-CTF + CyBench and reference tables are frozen. Published full-transcript direct judging covers 2025-II and 2026-I: **6,494 usable / 6,514 retained ratings**, across six judge/semester settings. See the [cyber judging README](../../logs/direct/cyber/README.md). Its results exports apply the final retained-input mask; nine additional valid R1 ratings in raw archives are outside that mask. Use the manifest or filtered exports, not an unfiltered concatenation of logs.
- **Coding:** project commit `4bbcfab399985674b64ac1516aca2257dcdcd126` records AgentBench OS 4-A **23/23 complete** and the expanded coding refreeze. The project pins `LASR-G5/benchmarks@d071b9e5630935e5e08c222a3b3002cbb6fc52a1`; the pinned revision and coding metadata files were verified on HF for this handoff. The previous “19/23, wait for refreeze” note is stale.
- **Coding judging uses the earlier freeze:** the [published direct campaign](../../logs/direct/coding/README.md) has **1,668/1,668 valid ratings**, but used benchmark revision `e3d86809e0980b968fd92add975db9fba6849ece`, before the AgentBench OS expansion. It does not establish full judging coverage of the new task. Load the expanded frozen tables, reconcile selected inputs, reuse exact matches, collect missing full-transcript judgments, and recalibrate against the matching θ* before comparing truncation on that freeze.

Start cyber independently of coding. Keep each full/truncated comparison on one frozen task, identical input eligibility rules and calibration cohort. Report exclusions and missing predictions. A nominal “full” baseline still has configured context limits; preserve its documented retained set rather than silently widening it.

Paraphrasing is optional and has not been collected. If included, preserve substantive content, final answers, code, tool calls and tool outputs. LLM compaction was in the original rough outline but has no separately agreed or collected condition; it is not a prerequisite for truncation.

## 4. Hidden reasoning: exact coverage and gaps

“Without reasoning” means **hide the subject's recorded reasoning from the protocol's inputs**. It does not mean turn off the subject's reasoning during generation, recollect subject rollouts, or disable the judge's own configured reasoning. Final answers remain available.

### Collected

**Pairwise:** the reasoning-off condition is already done, as agreed in the discussion. Reuse it rather than launching another sweep. Pairwise remains under `logs/judge/<domain>/<semester>/<judge>/<condition>/`; visibility is identified by the condition hash and metadata. The new `direct_reasoning-hidden/` root applies to direct judging only. Check the selected pairwise slice and its retained inputs when analysing it; this status is not a claim that every later domain refreeze is already judged. Pairwise's short-response filter can remove answer-only inputs, so report coverage alongside MSE.

**Direct:** collected for **every primary maths and science judge from 2025-I onward**, 17 conditions, published in `LASR-G5/judging@08dfef81` under `direct_reasoning-hidden/<domain>/<semester>/<judge>/<condition>/` (maths `4a2c79ff`, science `48cc07e2`). Each condition mirrors its visible campaign's frozen inputs, exclusions, references and target truth; inputs whose hidden rendering is byte-identical reuse the visible rating with verified provenance. 2024-I and 2024-II are not duplicated: no subject in those semesters recorded reasoning, so hidden equals visible for every retained input (recorded in `artifacts/direct-judging-hidden-reasoning/sweep-summary.json`).

| Domain | Conditions | Usable / retained | Missing |
| --- | --- | ---: | ---: |
| Maths | deepseek-v3, phi-4 (2025-I); deepseek-r1-0528, minimax-m1, qwen3-235b-a22b (2025-II); glm-4.5, glm-4.7, kimi-k2-thinking (2026-I) | 11,388 / 11,389 | 1 (GLM-4.7 cap failure, unchanged) |
| Science | deepseek-v3, phi-4, qwen2.5-72b-instruct (2025-I); deepseek-r1-0528, minimax-m1, qwen3-235b-a22b (2025-II); glm-4.7, kimi-k2-thinking, minimax-m2 (2026-I) | 13,569 / 13,569 | 0 |

The 15 new conditions made 15,492 new API calls at a recorded cost of $52. Empty completions and rate-limit gaps were recovered with same-config, cache-bypassed retries; failed task attempts that supplied cache origins, superseded originals and retries are archived under `legacy/direct_reasoning-hidden/`. The per-condition table is in `logs/direct_reasoning-hidden/README.md`; the publication record is `artifacts/direct-judging-hidden-reasoning/published-sweep.json`.

**Paired comparison** (`artifacts/direct-judging-hidden-reasoning/comparison/visibility-comparison.{csv,md}`, produced by `compare_visibility.py` from published logs, both conditions recalibrated on the same references, plus a matched-input variant): hiding reasoning lowers mean target ratings in 13 of 17 conditions, and calibration absorbs most of it. Matched-input MSE rises in 15 of 17 conditions, mostly by under 0.1 and at most by 0.8 apart from one outlier; the largest degradations are qwen2.5-72b science 2025-I (0.13 to 2.07), phi-4 maths improves (3.03 to 1.15) and phi-4 science improves (4.14 to 3.16); every 2026-I condition changes by under 0.07. All 17 hidden conditions forecast all 8 targets.

### Not yet collected or evaluated

| Area | What is absent | Required next step |
| --- | --- | --- |
| Legacy length baselines | A recorded reasoning-hidden baseline evaluation | Existing `chars`, `answer_chars`, `reasoning_chars`, `tokens` and `ridge` code can be reused, but apply the hidden view consistently before feature extraction. This is offline feature extraction/refitting, not subject collection. |
| New transcript baselines | End-to-end reasoning-hidden length/regex/structure evaluations | Length/regex extraction and `TranscriptRegression` exist in `src/protocols/transcript_features/`. The reader currently consumes raw structured messages without a reasoning-visibility option. Add the hidden view and its cache identity, and finish registry/evaluation integration. |
| Structure/RST baseline | RST clause labeller and the resulting annotations/evaluation | The structure variant requires a supplied `clause_typer`; that labeller is still a remaining implementation step. Generate any needed labels from the same permitted view for calibration and targets. This may require new annotation calls, depending on the chosen implementation. |
| Supplemental maths judges | Hidden conditions for Llama-3.3-70B (2025-I) and WizardLM-2-8x22B (2024-II), published to `direct/math/` after the sweep was scoped | Llama-3.3 needs a hidden run (2025-I inputs change); WizardLM 2024-II is identical by construction. `prepare.py` needs their campaign manifests (`artifacts/direct-judging-math-llama/`). |
| Earlier direct semesters | 2024-I / 2024-II | Verified identical: zero changed inputs in every 2024 condition, so the visible ratings are the hidden ratings. |
| Direct coding/cyber hidden reasoning | Hidden direct conditions for either agentic domain | Not collected. Their published full-transcript judging is separate from this ablation. Any expansion needs a matched freeze/baseline, especially coding. |

Do not describe all baseline code as “still unimplemented”: parts now exist. The outstanding work is completion/integration, enforcing the hidden view, any needed annotations, and evaluation. In particular, the legacy `tokens` feature counts billed output tokens **including reasoning**; it is not a reasoning-free observable. Drop or explicitly redefine such features for the hidden-view experiment. A judge's `include_reasoning=False` flag does not automatically affect either feature pipeline.

### Reproduction and implementation notes

- [Completion record](../../artifacts/direct-judging-hidden-reasoning/completion.json): final coverage, declared failure, publication commits and relocation.
- [Frozen collection scope](../../artifacts/direct-judging-hidden-reasoning/launch-ready.md): configs, counts and provenance. `inputs/<domain>/job.json` stores exact inputs, references, target truth and reuse ledger. Do not regenerate these to follow unrelated coding updates.
- The canonical hidden folders contain `ratings.csv`, `reference-scores.csv`, `forecasts.csv`, `metrics.csv`, `condition.json` and audited `.eval` logs. Original local `publication/` snapshots and `published-*.json` records intentionally retain paths valid at their original commits (`56305475` science, `817f0877` maths).
- The collection harness is under `artifacts/direct-judging-hidden-reasoning/`: `run.py`, `recollect.py`, `consolidate.py`, `audit.py`, `prepare_publication.py`. Completed raw attempts and retry logs are historical evidence, not active work. Prefer canonical published results over pre-repair exports.
- Sweep harness: every script in `artifacts/direct-judging-hidden-reasoning/` takes a condition `<domain>/<judge>/<semester>` (legacy `math` / `science` still address the GLM-4.7 inputs). `prepare.py` freezes all conditions from the visible manifests; `run.py`, `recollect.py`, `consolidate.py`, `audit.py --log-dir effective/<cond>`, `prepare_publication.py`, `render_readmes.py`, `publish.py`, `compare_visibility.py`. Lessons: the frozen `max_retries=2` is too low for Novita/SiliconFlow (use `--max-retries 10 --max-connections 8`); Inspect `eval_set` leaves error logs from failed task attempts beside the successful retry, so planners must consider only successful logs; a cache hit can originate in a cancelled attempt no log records, which `recollect.py --force task:item` re-requests; the HF partial clone cannot lazy-fetch single objects, so rebuild missing blobs from `/raw/` before fast-forwarding.
- Production use is explicit: `JudgeDirect(config, domain, include_reasoning=False)`. The constructor otherwise defaults to visible reasoning; the standard `--direct-judge` CLI does not expose this ablation. Setting only a YAML flag is insufficient with that constructor call. `DirectJudgeConfig` defaults to `log_dir=logs`, and the protocol selects `direct/` or `direct_reasoning-hidden/`.
- The protocol name/default forecast filename does not distinguish visibility. Keep analysis exports in condition-specific locations or give evaluation outputs distinct names so hidden results cannot overwrite visible results.
- The hidden-view constructor/path changes and related tests are local parent-repository edits at handoff; the HF folder move is committed and pushed. Preserve these edits when changing checkouts. The previous verification run passed 13 direct/historical tests; this documentation task did not rerun collection or tests.

## 5. Introduction of reasoning: one historical maths case

Use the existing **2025-I maths task**, calibrating on **2024-II references** and predicting the 2025-I reasoning and non-reasoning subjects. This is the agreed historical transition case, not a new sweep over hypothetical step changes.

Pairwise judging for this case is essentially complete. Direct DeepSeek-V3 and Phi-4 forecasts are available locally at `artifacts/direct-judging-math/results/<judge>/2025-I/forecasts.csv`; this handoff checked that each has all eight finite target forecasts. The [published maths campaign](../../logs/direct/math/README.md) also records both judges as complete. Llama-3.3-70B was omitted because its configured route was unavailable; that omission does not block the two-judge analysis.

Remaining work: assemble the existing predictions and frozen θ*, report MSE/coverage, and compare signed errors for early reasoning versus contemporaneous non-reasoning subjects. Check for systematic underprediction of the reasoning subjects. Label groups from the recorded model roster/configs. Do not expose target θ* during calibration. This analysis is independent of collecting hidden reasoning for every historical judge.

## 6. Style robustness

Dropped as a standalone challenge, including the distilled-model suggestion. Optional paraphrasing is covered under challenge 3. No separate collection backlog should be created for challenge 6.

## 7. Domain generalisation

This is **analysis-only when existing source evidence and destination labels overlap**. Choose A→B by shared reference/target coverage, preserving the temporal split and A's selected EASY inputs. Use B's task-specific HARD θ* for calibration and evaluation:

- **Direct:** reuse A transcript ratings; recalibrate earlier models' mean ratings against their B θ*.
- **Pairwise:** reuse A comparisons; refit Bradley–Terry with references anchored at their B θ*.
- **Transcript regression:** reuse A features; fit against earlier models' B θ*.

Target B labels are evaluation-only. Do not compare an unmodified A-scale θ̂ directly with B's θ*, and do not substitute another semester's destination labels when the required ones are missing. Keep model identity, task, available calibration labels and prediction coverage explicit.

Start with one well-covered pairing; maths/science→cyber is a useful candidate because cyber labels are already frozen. It does **not** require cyber judging: cyber is the destination. The original “new domain” task is non-agentic→agentic transfer, not prediction in a domain with no historical labels. Transfer to coding should use the expanded freeze consistently. New judging is needed only if the selected source evidence is missing; otherwise the work is joining, recalibration/refitting and reporting.

## Next actions and collection discipline

1. Done: paired reasoning-visible/hidden comparison for all 17 direct conditions (see section 4).
2. Complete the remaining baseline paths and their hidden-view evaluations; collect annotations only where that method requires them.
3. Analyse the historical maths transition and one domain-transfer pairing using existing evidence.
4. Select cyber truncation conditions and collect the changed-input judgments. Reconcile coding judging with the expanded freeze before adding coding truncation.
5. Remaining hidden direct extensions: supplemental Llama-3.3-70B maths 2025-I; coding/cyber hidden conditions if wanted.

For any later collection, follow **each judge's config** for API versus Runpod hosting, model, provider, precision, generation settings and permitted GPU fallbacks. GLM-4.7 hidden collection ran locally against its configured API (`atlas-cloud/fp8`, temperature 0, output cap 32,768); it did not require a model-hosting pod. The separate reasoning-visible WizardLM science backfill did require Runpod and finished 1,501/1,501, published in `0910026e`; it is not an outstanding hidden run.

The previous recovery pods were terminated and cleanup verified at completion. No live Runpod inventory was taken for this documentation task. Before resuming anything, inspect actual workers/resources; do not restart from historical launch instructions or leave idle/crashing pods billing. Load the project's `.env` with override when selecting the account, preserve failed attempts, audit before publication, and sync only intended results via Git to the independent `logs/` HF repository. Do not upload entire artifact directories.
