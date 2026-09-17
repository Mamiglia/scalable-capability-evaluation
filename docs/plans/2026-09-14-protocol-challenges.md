# Protocol challenges

**Current operational status:** see the [2026-09-15 handoff](2026-09-15-protocol-challenges-HANDOFF.md), including the completed coding refreeze, published cyber/coding judging, and exact reasoning-hidden coverage and remaining work. Collection-status snapshots below are historical where superseded by that handoff.

Discussion draft, updated 2026-09-15. Models, judges, evaluation subsets, and perturbation settings remain to be selected where noted. This document records the experimental scope, dependencies, and collection status.

## Existing core: future prediction on harder items

Original challenges 1 and 2 are already built into the project; they are not additional experiments to implement.

Protocols are calibrated on earlier models and predict later models' domain-specific HARD-item capability, θ*, from their correct, first-attempt transcripts on selected EASY items. The primary metric remains MSE between predicted capability and θ*.

The challenges below retain this future-to-hard setup. Changes to transcript presentation leave the underlying subject rollouts and θ* unchanged. Reuse existing data and matching predictions wherever possible; these challenges do not require rerunning every model, judge, domain, and semester.

## Work and dependencies

| Challenge | Work required | Dependency |
| --- | --- | --- |
| Truncation | New judgments/predictions on truncated coding/cyber transcripts | Cyber is frozen and ready for preparation; coding awaits the AgentBench OS completion/refreeze |
| Hidden reasoning | Pairwise done; direct maths/science collected, audited, and published; other baselines remain | Remaining baseline approaches still need implementation |
| Reasoning introduction | Analyse the historical maths transition using existing forecasts | Pairwise judging essentially complete; two direct judges already have completed transition results |
| Style robustness | Drop as a standalone challenge | Paraphrasing can remain an optional transcript condition |
| Domain generalisation | Recalibrate/refit using existing source-domain evidence and destination labels | Overlapping models, source judgments, and destination HARD-item θ* |

## 3. Truncation: cyber ready, expanded coding nearly ready

**Question:** Does capability prediction still work with truncated agentic transcripts?

Restrict truncation to coding and cyber. HF update checked at `e3d86809e0980b968fd92add975db9fba6849ece` on 2026-09-14:

- **Cyber:** collection and reference freezing are complete. The 117-item bank combines InterCode-CTF and CyBench. The 2025-II task has 70 EASY/train and 47 HARD items; 2026-I has 79 EASY/train and 38 HARD items. The collection note says judging has not started. Truncation preparation can begin now, including a matched untruncated judging condition.
- **Coding:** the existing SWE-bench mini / Terminal-Bench bank is already frozen. The pending addition is AgentBench OS 4-A: 19/23 models are pushed, with four still outstanding or being repaired. The collection plan calls for refreezing coding with AgentBench OS once those finish. Use that expanded freeze for the planned coding truncation test.

Cyber does not need to wait for coding. Keep the corresponding frozen EASY/HARD split and θ* fixed within each truncation comparison. Metadata and collection notes are recorded under `artifacts/protocol-challenges/hf-status/`; no full rollout logs were downloaded for this status check.

Then select a small set of tasks and judges, and compare original transcripts with deterministic prefix truncation at several lengths. This already exists through `max_length`. Reuse subject rollouts and HARD-item θ*; only the affected judgments and protocol predictions need to be recomputed.

**Agreed:** apply the same truncation rule and length limit to both reference/calibration and target transcripts. Calibration and prediction therefore use the same viewing limits.

Paraphrasing can remain an optional secondary condition, preserving substantive content, answers, code, tool calls, and tool outputs. It is not a separate style-robustness campaign. Short-task-to-long-task prediction is outside the current scope.

## 4. Hidden reasoning

**Question:** Does the protocol work with reasoning excluded from the visible transcript?

**Pairwise:** the reasoning-off condition is already done. Reuse those results; no additional pairwise sweep is proposed.

**Direct:** a limited maths/science comparison is worth running without waiting for coding/cyber collection. This requires new judgments where the visible input changes, but no new subject rollouts or IRT fits.

- Current scope: GLM-4.7 on 2026-I in maths and science, paired with the corresponding reasoning-visible baseline.
- Exclude reasoning from both calibration and target transcripts. Keep the same calibration cohort, items, prompt, and fitting method as the baseline.
- Reuse existing ratings when stripping reasoning leaves the rendered input unchanged, after checking input and judging-setting equivalence.
- Direct judging now accepts `JudgeDirect(..., include_reasoning=False)` for this condition, retaining the reasoning-visible default for existing campaigns.

This is a limited additional direct condition. Report MSE and prediction coverage on matched subjects/items. Existing pairwise short-response exclusions should remain visible when interpreting its results.

**Collected, audited, and published, 2026-09-15:** GLM-4.7 on 2026-I maths/science. Science has 1,538/1,538 usable ratings, including 26 verified identical-input reuses. Maths has 1,450/1,451 usable ratings, including five reuses; seven initially unusable ratings were recovered. One Apex-shortlist judgment exhausted the configured output cap on its original request and both retries and remains explicitly missing. The original and retry logs are preserved.

Current HF folders: `direct_reasoning-hidden/science/2026-I/glm-4.7/48cc07e2/` and `direct_reasoning-hidden/math/2026-I/glm-4.7/4a2c79ff/`, alongside `direct/` for reasoning-visible runs. Archived attempts mirror the new root under `legacy/direct_reasoning-hidden/`. Both hidden conditions retain `include_reasoning: false`, their original condition hashes, and the `reasoning_hidden` label. Original publication commits `56305475` and `817f0877` retain the earlier layout; the folder move does not change judgments or scores. Final local publication snapshots are under `artifacts/direct-judging-hidden-reasoning/publication/`.

**Other baselines:** implementation/integration and reasoning-hidden evaluation remain; some feature code now exists (see the handoff). Evaluate them with reasoning excluded from both calibration and target inputs, using the same tasks and HARD-item θ*. The limited direct condition is collected; its paired analysis and the remaining baseline work are outstanding.

## 5. Introduction of reasoning: maths only

**Question:** Do protocols predict the capability increase associated with the introduction of reasoning models?

This is one historical case, restricted to maths. Use the existing 2025-I task: calibration on 2024-II references and prediction of the 2025-I cohort containing reasoning and non-reasoning subjects.

- Retain calibration on models from before the transition.
- Evaluate early reasoning models and contemporaneous non-reasoning models.
- Compare predicted capability with maths HARD-item θ*.
- Inspect whether reasoning models are systematically underpredicted relative to non-reasoning models, alongside the usual MSE.

**Pairwise judging is essentially complete for this historical case.** The remaining work is to analyse its existing forecasts against frozen θ*, not to launch a separate judging campaign. Direct results support the same analysis as they complete.

**Already available locally, checked 2026-09-14:** direct maths results for DeepSeek-V3 and Phi-4 on 2025-I are marked completed, with forecasts for all eight targets. They are under `artifacts/direct-judging-math/results/<judge>/2025-I/`. The direct case can therefore be analysed already; matching pairwise results should be read from the existing judging outputs. These files establish data availability, not a completed challenge analysis.

## 6. Style robustness: dropped as a standalone challenge

Drop the separate style-robustness and distilled-model experiments from the current plan. Paraphrasing, if retained, belongs under transcript transformations in challenge 3.

## 7. Domain generalisation

**Question:** Can transcripts from one domain predict HARD-item capability in another domain, while retaining the temporal split?

This is analysis-only when existing source judgments/features and destination θ* labels cover the same subjects and calibration models. Retain the temporal split and source EASY-item selection.

For source domain A and destination domain B:

- **Direct:** reuse ratings of A transcripts and recalibrate them against earlier models' B HARD-item θ*.
- **Pairwise:** reuse comparisons of A transcripts and refit Bradley–Terry with reference models anchored at their B HARD-item θ*.
- **Transcript regression:** reuse A features and fit against earlier models' B HARD-item θ*.
- Predict later models' B capability and evaluate against their B θ*. Target labels remain evaluation-only.

The destination calibration/refit is necessary: source-domain capability predictions cannot simply be scored against a different domain's θ*. No new judging is needed for existing source inputs and comparisons with adequate coverage.

Original task 7a covers domain transfer generally. Task 7b focuses on non-agentic-to-agentic transfer, such as maths or science transcripts predicting coding or cyber capability.

**Proposed starting scope:** one non-agentic-to-agentic pairing, selected according to overlapping model coverage. Cyber destination labels are now frozen, so maths/science-to-cyber analysis can proceed from existing source judgments after checking shared model coverage. It does not require cyber judging. Transfer to the expanded coding bank should use the forthcoming AgentBench OS refreeze. Maths/science transfer can also use existing frozen labels and completed source judgments.

This proposal assumes destination-domain labels are available for earlier models. Prediction in a domain with no historical labels would be a separate question.

## Next steps

- Compare the audited, published reasoning-off direct maths/science results against their reasoning-visible baselines, reporting the declared missing rating and coverage.
- Implement the remaining baseline approaches and include their reasoning-off evaluations in the hidden-reasoning comparison.
- Analyse the historical maths case using the essentially completed pairwise judging and available direct forecasts; incorporate any remaining results as they complete.
- Prepare maths/science-to-cyber analysis from existing judgments and the newly frozen cyber labels, checking overlapping model coverage.
- Choose cyber truncation subjects, judges, and limits, with matched full-transcript judgments.
- After AgentBench OS collection and the expanded coding refreeze, add coding truncation and transfer analyses.

Keep comparisons paired on the same subjects where possible, and report the number of targets with valid predictions alongside MSE.
