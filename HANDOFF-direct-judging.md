# Direct judging collection handoff

## Latest completion update — 2026-09-15

- WizardLM science is complete: 1,501/1,501 valid ratings, 32 tasks. It followed the configured Runpod BF16 deployment and used the allowed four-H100-NVL fallback after H200 startup failures. All recovery pods were terminated. Audit and publication records: `artifacts/direct-judging-science/recovery-20260915/`. HF commit `0910026e`.
- GLM-4.7 reasoning-hidden 2026-I runs are audited and published: science 1,538/1,538 ratings (`48cc07e2`, commit `56305475`); maths 1,450/1,451 (`4a2c79ff`, commit `817f0877`). Seven maths ratings were recovered; one remains missing after repeated configured-cap exhaustion. Original/retry traces are preserved in the HF legacy area.
- The new reasoning-hidden conditions have `include_reasoning: false` and the `reasoning_hidden` metadata/tag. Judge reasoning itself follows the model config.
- Current HF storage separates these under `direct_reasoning-hidden/`, alongside `direct/` for reasoning-visible runs. Hidden-run archives are under `legacy/direct_reasoning-hidden/`; hashes and raw data are unchanged.
- These completion records supersede the running/failed process information in the older handoff below. Do not restart completed workers from those old instructions.

Updated 2026-09-14, approximately 17:02 UTC (18:02 London). Live JSON files supersede the counts below.

## Immediate state and next actions

1. **Maths is running.** Latest checked progress: 9,058 / 16,582 responses saved; 9,025 parsed ratings. Continue the existing supervisor; do not launch a duplicate campaign.
2. **WizardLM science has restarted using its original configured worker.** Worker PID `62009`, wrapper PID `62008`, exec session `19867`. Runpod has allocated `jjekc41yyrnsdw`, 4 × H200, $18.36/hour. At this update it has no public IP and no judgments: allocation is confirmed, serving is not. Check startup and actual judgments next.
3. **Science MiniMax-M1 is still running**, PID `29009`. At approximately 17:00 UTC it had 1,180 / 1,520 saved responses. Its original supervisor imposes a three-hour worker deadline around **17:14 UTC**. If it times out, preserve failed logs before resuming. Maths M1 waits for this process to exit, then may start automatically; coordinate any science resume to avoid exceeding the configured shared 16 connections.
4. Science had 20,504 / 22,332 responses saved (includes 4,380 reused pilot judgments), 20,295 parsed. Thirteen judges have completed; M1 and Wizard remain. Saved responses are not synonymous with valid parsed ratings.
5. After collection, audit missing/invalid ratings and actual provider provenance, then produce final results. No campaign outputs have been pushed to HF.

## User decisions and authorization

- Work in this checkout, not a worktree. Do not inspect sibling `small-judge` except for pricing. Use this project's `.venv/bin/python`.
- Science collection, its configured OpenRouter routes, transcript reasoning disclosure, and configured Runpod judges were explicitly approved. User subsequently requested maths collection and the Wizard restart. No new approval is needed for those existing operations.
- Follow the model configs, including provider pins, generation settings, precision and configured GPU fallbacks. Do not substitute an unavailable API provider.
- One to three judges per semester, chosen from the previous semester's frontier models. Do not run all judges for all semesters.
- Judge only correct first-attempt transcripts from the frozen EASY item set. This is not pairwise's both-correct eligibility rule.
- Judges see one transcript at a time, including the subject's available reasoning and its original options. No silent truncation: skip context-overflow and character-cap cases.
- Preserve decimal scores on the 1–10 scale. Average transcript ratings per model, then fit linear calibration to the reference capability labels.
- **Primary analysis uses previous-semester references only.** All-history calibration is a separate future ablation. Never substitute global or another semester's theta for current-task HARD theta.
- User accepted the strong question-difficulty dependence as a method limitation. Do not change the prompt, anchoring or calibration during collection.
- Do not download full HF logs locally. Only metadata and bounded ranges were fetched.
- HF publication uses git only, never `hf upload`. Do not send messages to teammates without authorization.

## Credentials and security

**The inherited `RUNPOD_API_KEY` belongs to a different account.** Always load the project `.env` with `override=True` before project API/CLI operations:

```python
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path.cwd() / '.env', override=True)
```

Do not print keys. The other account's `gemma-core-scoring` CPU pod is unrelated; do not touch it. Workers remove `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` before provisioning these public-weight Runpod judges.

An unnecessary SSH diagnostic detour attempted an account-level SSH-key addition. Automatic approval review rejected that persistent security change before execution; **no account key was added**. Do not bypass that rejection. The active restart uses the ordinary configured worker and needs no SSH changes.

`artifacts/direct-judging-science/wizard-recovery/id_ed25519` is a local private key from the abandoned diagnostic attempt. **Never upload the entire artifact directory or that key.** The old diagnostic pod `3tfoc2ti5o7agz` was terminated; its recovery logs are not current status.

## Live files and processes

Campaign roots:

- `artifacts/direct-judging-science/`
- `artifacts/direct-judging-math/`

Each contains `progress.json`, `supervisor.log`, `run-manifest.json`, `workers/<judge>.json`, isolated `runpod-sessions.json`, configs, source manifests, raw `.eval` logs, Inspect caches and `results/<judge>/<semester>/`.

Results include `ratings.csv`, `reference-scores.csv`, `forecasts.csv`, `metrics.csv`, and `result.json`. Check these and actual logs rather than relying exclusively on supervisor exit status.

| Process | PID / exec session | Notes |
| --- | --- | --- |
| Science original supervisor | PID 27742 / session 11842 | Old Wizard failed entry is stale for the manual restart. |
| Science M1 | PID 29009 | Original three-hour deadline around 17:14 UTC. |
| Maths supervisor | PID 58621 / session 26326 | Three API workers plus one GPU worker; M1 waits for science M1. |
| Maths Qwen2-72B | PID 61382 | Allocated pod `37l07trz4aaige`, 4 × A40, $1.96/hour; startup pending at latest check. |
| Wizard restart wrapper / worker | PIDs 62008 / 62009, session 19867 | Started 16:58 UTC; three-hour timeout includes startup. |

Wizard-specific live files are `wizard-restart.json`, `wizard-restart.log`, and `workers/wizardlm-2-8x22b-hf.json` under the science root. The old supervisor will still report Wizard's earlier exit `-2`; this does not mean the restarted worker is paused. The Runpod session registry can temporarily say `creating` even after the API has allocated a pod, while `runpodctl pod create` is waiting.

Latest direct API checks confirmed both new pods exist with desired status `RUNNING`, but neither has public IP, port mapping or runtime metadata yet. Wizard's log endpoint returned `404 pod not found` while pod get/list succeeded; this is not evidence that the pod has been deleted. No startup log or serving health confirmation is available yet. If a worker stops before `create` returns its pod ID, verify cleanup against the actual IDs above rather than assuming the registry knows them.

Both supervisors were launched with `--budget-usd 300 --worker-timeout-hours 3`. Budget is an estimate from saved API token usage, **not a hard billing cap**; it excludes GPUs, in-flight calls, hidden retries and fees. Science's most recent new API estimate was about $58.3; maths was above $33.8 and increasing. Do not conflate these with total billed cost.

## Commands and reproducibility

Run commands from the project root. Active temporary runners:

```sh
PYTHONPATH=src .venv/bin/python /tmp/direct-judging-science/run_science.py --self-test
PYTHONPATH=src .venv/bin/python /tmp/direct-judging-math/run_math.py --self-test
```

The science self-test passed after the latest harness-only `retry_cleanup=False` edit. Maths self-tests and full production-render checks passed before launch.

Existing launches, recorded for recovery only (do not run while their supervisors are alive):

```sh
PYTHONPATH=src .venv/bin/python /tmp/direct-judging-science/run_science.py --run --budget-usd 300 --worker-timeout-hours 3
PYTHONPATH=src .venv/bin/python /tmp/direct-judging-math/run_math.py --run --budget-usd 300 --worker-timeout-hours 3
```

Single-judge recovery uses the matching runner with `--judge NAME`; this mode has no supervisor lifetime cap by itself. The Wizard wrapper `/tmp/direct-judging-science/restart_wizard.py` adds a three-hour timeout and SIGINT cleanup. Always check processes, registry and existing pods before restarting. Prefer graceful SIGINT so logs flush and the model context manager terminates its pod; verify deletion afterward.

Runner and preparation script copies are saved in each campaign's `handoff-scripts/` directory. The original `/tmp` files remain the active versions. Manifests hash the prompt, configs and source inputs; do not regenerate or modify the frozen campaign inputs during collection. Reuse successful tasks and Inspect cache entries, preserving failed/cancelled logs. Current runner resumes set `retry_cleanup=False`.

## Science sources and roster

HF source: `LASR-G5/benchmarks` at `6620467bb4391ff8f0ab3b2672ad65597a7d2295`.

- Frozen science models/items/fit are byte-identical to the pilot's. Repaired Gemma-4-31B GPQA source changed, so affected judgments were recollected.
- 6,960 unique selected first-attempt source records, 7,478 correct semester uses. Only 52.44 MB fetched as ranges versus 926.2 MB full logs; local subsets approximately 31.58 MB.
- 4,380 pilot judgments reused only after exact input/system/settings verification. Old pilot copies remain in `legacy/pilot/` with provenance.
- Current retained target: **22,332**, including reuse, after 102 excluded uses. The original running supervisor still has the old DeepSeek-V3 count of 1,498 in memory; authoritative revised manifest/result count is 1,497.

| Semester | Judges |
| --- | --- |
| 2024-I | GPT-4-0613, Mixtral-8×7B, Nous-Hermes-2-Yi-34B |
| 2024-II | GPT-4o-2024-05-13, GPT-4 Turbo, WizardLM-2-8×22B-HF |
| 2025-I | DeepSeek-V3, Phi-4, Qwen2.5-72B |
| 2025-II | MiniMax-M1, DeepSeek-R1-0528, Qwen3-235B |
| 2026-I | Kimi-K2-Thinking, GLM-4.7, MiniMax-M2 |

WizardLM-HF **is still rostered** in the latest checked remote configs and frozen HF fit. The old hosted `wizardlm-2-8x22b` is the dropped variant. User clarified that a teammate likely terminated/stopped the earlier run; do not state that its disappearance proved a model/vLLM fault.

Wizard config: public `alpindale/WizardLM-2-8x22B`, `vllm/vllm-openai:v0.28.0`, BF16, tensor parallel 4, max model length 16,384, configured output 12,000, 400 GB container disk. H200 primary, then configured H200 NVL / H100 NVL / RTX PRO 6000 Blackwell fallbacks. Current restart obtained H200.

## Maths sources and roster

HF source: `LASR-G5/benchmarks` at `7352568520df9ea565e9e1c85d9af14889813c88`, including repaired Gemma AIME24/25 logs from 2026-09-14.

- All 95 selected Gemma records across six benchmarks have reasoning; repaired AIME24 9/9 and AIME25 7/7 verified typed reasoning.
- 4,751 unique correct first-attempt records, 6,098 semester uses. 74.67 MB range transfer versus 916.31 MB full logs; approximately 44.74 MB local subsets.
- Frozen train item counts: 33 in 2024-I, 95 in 2024-II, 100 thereafter.
- 14 available judges, 732 tasks, **16,582 retained judgments**: 14,851 API and 1,731 Runpod.
- 225 exclusions: Phi-4 70, R1 140, and five over-character-cap records for each of the three latest judges (15).
- Llama-3.3-70B's configured `crusoe/bf16` route is absent: omitted 1,487 judgments. No provider substitution; remaining two judges run for that semester. User was informed.

| Semester | Judges |
| --- | --- |
| 2024-I | GPT-4-0613, GPT-3.5-Turbo-0125, Mixtral-8×7B |
| 2024-II | GPT-4o-2024-05-13, GPT-4 Turbo, Qwen2-72B |
| 2025-I | DeepSeek-V3, Phi-4; Llama-3.3-70B omitted (route unavailable) |
| 2025-II | DeepSeek-R1-0528, MiniMax-M1, Qwen3-235B |
| 2026-I | Kimi-K2-Thinking, GLM-4.7, GLM-4.5 |

Math subsets store **only correct** records. `run_math.py` patches transcript reads to stream these local subsets and filter requested IDs; direct indexed reads of all train IDs would fail. Paths absent from `semester-inputs.csv` return an empty set (no correct records); listed paths must exist and read successfully. `ReferenceJudgeDirect` explicitly restricts primary training to references even when historical labels become available.

## Known issues to preserve in the audit

**DeepSeek-V3 science: recovered and complete.** One GPQA/Qwen3-32B item (`recQ8oxSY2aCwFRS8`) hit an undocumented backend prefill limit: actual 40,583 prompt tokens exceeded 32,768 despite the advertised 163,840 context. Added exactly one exclusion. Finished 1,497/1,497 valid; recovery used cache, no new paid calls. Thirty-one other successful task archives stayed byte-identical. Maths inputs were screened against this observed limit and fit.

Inspect's default retry cleanup deleted the earlier cancelled 21-sample DeepSeek archive. **Do not claim that archive was preserved.** The original gzipped trace, all 13 timestamped errors, input reconstruction, hashes and pre-resume ratings were recovered under `deepseek-recovery/`; see `archive-audit.json` and `canonical-completion.json`. Future resumes disable retry cleanup and should pre-copy failed logs.

**Mixtral parsing: pending offline repair/audit.** Science has 1,420 saved / 1,282 parsed; maths 417 / 386. Science's 138 rejected outputs generally contain valid numbers followed by prose, e.g. `ANSWER: 7. The model...` or `ANSWER: 6 (explanation...)`. The strict parser requires the entire ANSWER line to be numeric. Raw outputs are retained. Prefer a consistently applied, tested offline rescoring change over new paid calls; no parser change has yet been made. Also inspect M1 and other judges' invalid responses separately rather than assuming the same cause.

## Code, environment and publication

- Current checkout HEAD: `326afeecdda72adeb4a59f78bf30849e4d264b5c` (historical calibration PR #24 merge). Last checked remote methodology additionally had `46d6aee`, which hashes the item subset into task versions. Do not pull into active campaigns without checking source/manifest effects.
- Direct judging PR #23 was merged at `c91f14a`: <https://github.com/Mamiglia/small-judge/pull/23>. Description footer credits OpenAI Codex; no co-author history rewrite.
- Existing unrelated user changes must be preserved: modified `src/protocols/features.py`, untracked `src/protocols/transcript_features/`, and `tests/test_transcript_features.py`.
- Production direct implementation lives in `src/protocols/direct_judge/`; shared prompt `assets/system_prompt/direct.txt`, inline user prompt `src/judge/dataset.py`, single direct test file `tests/test_direct_judge.py`.
- `task.references` means previous semester, `task.historical` older models with finite current HARD theta, `task.train` their union. Generic supervised protocols now use train; pairwise overrides references. Frozen science currently has eight references and zero historical models in every semester, so its current harness still trains on references only.
- Interpreter was synced to the dependency declarations and overrides on remote main using `uv sync --inexact --no-dev`; main has no lockfile. Python 3.14.6, Inspect AI 0.3.263. Full inventory in science `environment.json`.
- 34 focused tests passed after direct PR merge. Broader run: 113 passed, seven failed, one collection error: Linux `prctl` on macOS (six), changed GPQA shuffle default (one), and upstream HLE removed import (collection). Frozen-table collection does not import the problematic response-loading path.
- `LASR-G5/benchmarks` and `LASR-G5/judging` are separate HF repositories. `data/` pins benchmarks; local `logs/` is an ignored independent judging checkout. Intended direct publication prefix: `judge/direct/<domain>/<semester>/<judge>/<condition>/*.eval` in `LASR-G5/judging`. Publish via git only and advance any parent pin after pushing. **Nothing from these campaigns is published yet.**

## Earlier diagnostics

Pilot and plots: `artifacts/direct-judging-pilot/science-2026-I/`. Current figure is `difficulty-diagnostic/difficulty-capability-densities.{png,svg,pdf}`, regenerated by its `plot.py`, using the frozen 4,505-rating snapshot. Six panels; continuous capability adds approximately 0.8–1.5 percentage points beyond item effects. Quartile density margins attach directly to panels c/e. User disliked purple colour schemes and distant/full-size histogram panels. Read the nature-figures skill before changing research plots.

The completed pilot's reference-only item centering worsened future-model prediction. User accepted this as a limitation, so current collection deliberately retains the existing prompt and raw-mean linear calibration.
