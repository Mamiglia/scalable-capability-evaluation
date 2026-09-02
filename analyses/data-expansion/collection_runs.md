# collection run log

## 2026-08-12 eval-variation second-wave tone judging + SWC handover (DONE)
- deepseek/qwen3-32b/glm judged confident+matteroffact (resume skipped the near-free three): full 6-judge tone columns. Cost $17.92 vs ~$15 quoted (glm output tokens again); week total ~$149.
- explanation mining (explanations.csv): length/detail mentions 61% of baseline explanations -> 43-49% standardized; correctness mentions up in pooled (.32->.36).
- figures 1-4 in judge_results/eval_variation/figures/ (pick-stronger, pick-longer, pool size, accuracy guard).
- SWC: repo+configs+215M data staged to $BASE/code/eval_variation/ (weights/venv/vllm reused read-only); pilot job 3383541 submitted (user authorized Claude-driven sbatch from remote control); grid follows pilot pass.

## 2026-08-11 eval-variation complete (tone collections + gap fill + tone judging)
- confident + matteroffact collections (7 models x 2 epochs each): 7/7 success both, accuracy guards clean (worst +-.045), lengths track baseline - tone isolated from length. Gemini rescores: confident .000->.745, matteroffact .038->.725.
- gap fill (sequential, counter-checked): qwen3-32b words50-100 remainder + full pooled; vl-30b words50-100 + pooled. qwen3-8b dropped from ALL paid judging - its columns come from the SWC re-run uniformly (avoids api-vs-vllm serving mix in same-judge comparisons).
- tone judging: gemini/ling/vl-30b on confident (422 matched items) + matteroffact (516), <$1.
- final matrix + findings: docs/2026-08-11-eval-variation-findings.md. Verdicts csv: judge_results/eval_variation/verdicts.csv (~57k rows).
- totals: ~$131 this week all-in (collections ~$42, judging round 1 $68 incl the overrun, gap fill ~$11, tone batch ~$22 of $30). Parked: deepseek/qwen3-32b tone ~$7, glm tone ~$8, SWC re-run free.

## 2026-08-11 eval-variation API judging (STOPPED at budget overrun)
- 3 conditions x cheap-7-pinned judges via analyses/eval-variation/run.py, +-50-word length filter at task build. Complete: words200 all 7 judges (1664 verdicts each); words50-100 deepseek/gemini/glm/ling complete + qwen3-32b partial 2070/2630; lenmatch-pooled deepseek/gemini/glm/ling complete (4636 verdicts each). Parse rate 99.8% on spot checks. Resumable: rerun the same configs, success logs skip.
- OVERRUN: authorized $50, spent $68.13 (counter-verified). Causes, in order: pooled condition was 4636 rows/judge not the assumed ~2630 (never sized before launch); glm-4.6 output tokens at $2/M ate ~$40+ (reasoning judges emit 2-3k out-tokens/verdict); three fleets ran in PARALLEL so the per-block tripwire never had a chance to fire; pinned-endpoint prices exceed the openrouter list prices used for estimates. Fleets killed on discovery.
- remaining gaps (qwen3-8b + vl-30b on w50-100/pooled, qwen3-32b remainder) can be filled free on the SWC fleet or on user say-so only.

## 2026-08-10 eval-variation words50-100 collection (DONE)
- same command shape as words200, bench gpqa_words50_100.yml ("Explain your reasoning in 50-100 words."), 2 epochs, logs data/eval-variation/gpqa-words50-100/ab96a27b/. 7/7 success, 396 samples each.
- target = deepseek's natural band (baseline p25-p75 49-87): compression is the direction models comply with. Medians: deepseek 53, glm 60, qwen3-8b 88, gemini 94, ling 107, qwen3-32b 116 all in/near band; vl-30b 188 REFUSES heavy compression (the new outlier).
- accuracy guard (base -> w50-100): think-mode holds (deepseek .905->.902, glm .778->.778 - its words200 dip was the expansion, qwen3-32b .629->.619, qwen3-8b .577->.545); non-think visible-CoT models pay for real (gemini .731->.669 rescored, stock read .018; ling .626->.482). vl-30b .552->.563 unchanged but only because it ignored the instruction - non-compliance protects capability.
- length-matched pool (50-word criterion): 1315 items vs 832 (words200) vs 386 (baseline). Deepseek pairs finally populated (160/97/80/50/49); vl-30b pairs now the empty ones (0-41).

## 2026-08-10 eval-variation words200 collection (DONE)
- command: collect -b analyses/eval-variation/bench/gpqa_words200.yml -m <7 cheap subject ymls> --log-dir data/eval-variation. System message "Explain your reasoning in approximately 200 words." via the gpqa_instructed wrapper task; 2 epochs (vs baseline's 4, cost cut); logs data/eval-variation/gpqa-words200/38c54e19/.
- 7/7 success, 396 samples each, pinned providers served all real generations (429 retries excluded). Instruction wording chosen after smokes: "150-200 words" range let ling drift under the floor; "approximately 200" clustered best. 200 ~= pooled median of baseline correct-answer words (per-model medians 67-440).
- compliance (median words, baseline -> words200): compression works, expansion doesn't. gemini 277->192, vl-30b 258->208, ling 440->222, qwen3-32b 418->237, qwen3-8b 305->162 all comply; glm 111->141 partial; deepseek 67->66 ignores it (reasoning never mentions the budget - think-mode final stays a summary).
- accuracy guard (delta vs baseline): deepseek +.004, qwen3-8b -.001, vl +.014, qwen3-32b -.018, gemini -.029 (rescored copy at .../gemini-3.1-flash-lite-rescored/, stock scorer read .010), glm -.058, ling -.081. The two biggest compressors pay real accuracy - the instruction is not free.
- length-matched pool (both-correct items within 50 words, all 21 pairs): baseline 386 -> words200 832. Deepseek pairs stay near-empty either way (max 7 except vs glm 56).
- qwen3-8b x {gpqa, aime2026}, limit 5, ~$0.04. Verified: alibaba/fp8 pin on every call, reasoning in logs, epochs bug found+fixed (task.py epochs default 1 -> 0). Logs moved to scratchpad, not kept.
- judge smoke: ling-2.6-flash on qwen3-8b self-pair, gpqa (8 verdicts) + aime2026 (6 verdicts), novita pin verified, prompts reasoning-free. Dumps in docs/tests/.

## 2026-08-08 cheap-6 collection, attempt 1 (ABORTED - seed bug)
- command: uv run collect -b configs/bench/gpqa.yml configs/bench/aime2025.yml configs/bench/aime2026.yml -m configs/models/qwen3-8b.yml configs/models/qwen3-32b.yml configs/models/qwen3-vl-30b-a3b-instruct.yml configs/models/glm-4.6.yml configs/models/ling-2.6-flash.yml configs/models/deepseek-v4-flash-0731.yml
- full runs (no limit), epochs 4 (gpqa via task_args, aime via yml), 18 jobs in parallel
- endpoint snapshot: data/openrouter_endpoints_2026-08-08.json (all 11 models)
- deferred to after top-up/cap-reset: gemini-3.1-flash-lite (GPQA maybe reused from data/gpqa/...WFQWWYCNt5pqBXa8EhkNLz.eval), gpt-4.1, o4-mini, claude-sonnet-4.5, claude-opus-4.7
- ABORTED ~40min in: collect passed seed=repeat to every call. deepseek/fp8 + siliconflow/fp8 (x2) don't support seed -> require_parameters excluded them -> 404 on all deepseek-v4/qwen3-32b/qwen3-vl jobs. Endpoints that DO support seed honored it intermittently -> some byte-identical epochs (seen in smoke: 3/4 epochs identical on one item). Fix: dropped seed from collect entirely; temperature sampling provides epoch diversity. Seeded partial logs moved out of data/ (scratchpad), not kept.

## 2026-08-08 cheap-6 collection, attempt 2 (DONE, seedless)
- same command and scope as attempt 1, after removing seed=repeat from src/collect/cli.py
- deepseek re-pin mid-run: first-party deepseek/fp8 is a 'paid model training' endpoint, data_collection deny excludes it (404). Re-pinned to novita/fp8 (same quant, 1M ctx); error logs removed, deepseek jobs relaunched separately.
- COMPLETE: 18/18 success. Verified per log: expected sample counts (792 gpqa / 120 aime), pinned provider served every real generation (429 retries excluded), reasoning present iff expected, zero config problems.
- accuracies (gpqa/aime25/aime26): deepseek .905/.992/.942, glm .778/.617/.600, qwen3-32b .629/.292/.325, ling .626/.542/.642, qwen3-8b .577/.158/.183, vl-30b .552/.533/.617
- notes: qwen3-32b hit length-stops on AIME (33+26 truncated epochs; lost epochs only, handled by both-correct filter). Some byte-identical duplicate epochs without seed (worst: qwen3-8b gpqa 67/792) - natural low-temperature convergence, noted for analysis.
- cost: ~$66 daily spend. Remaining: $77 credits, $72 weekly cap. Closed-5 collection + judging still needs the top-up.

## 2026-08-08 cheap-6 judging + sonnet/gemini collection (STOPPED - weekly key cap)
- judge run: configs/judge_cheap6_{gpqa,aime2025,aime2026}.yaml (45/396 comparisons, all-cheap judge+members), log_root logs/data-expansion/<bench>. Resumable - re-run the same commands.
- done before the 403 cap: gpqa 12/90 phase-runs (480 verdicts), aime2025 29/74 (984), aime2026 31/74 (988). Parse failures 4/988 on aime2026 spot-check. Judge pins verified (Novita/AtlasCloud/SiliconFlow/Alibaba only).
- gemini-3.1-flash-lite collection: DONE all 3 benches (fresh, seedless). claude-sonnet-4.5: gpqa errored at cap, aime partials killed - re-run collect for sonnet after reset/top-up.
- key hit its $500 weekly limit ~12:08 local. Resets Mon 00:00 UTC; teammate can raise per-key limit sooner. ~$75 credits remain (credits were not the binding limit).
- early read (partial, biased sample): weak judges near chance on non-self pairs (qwen3-vl .58, qwen3-8b .53, ling .50, qwen3-32b .50). Self-pair rows not yet reached.

## 2026-08-08/09 judging complete (cheap scope) + gemini fix + big-gap probe
- ALL judge logs: 290 success phase-runs, 22,922 verdicts, 0.4% unparsed, only pinned providers served.
- gemini gpqa scoring bug: model drifted to 'ANSWER: $C$' (pilot log parsed fine at .72) -> stock scorer read '' -> measured acc .004. Rescored copy (tolerant regex) at data/gpqa/63793eb6/gemini-3.1-flash-lite-rescored/ = .731. Original log untouched. FULL-RUN TODO: transcripts map must point gemini gpqa at the rescored log; tell maintainer the scorer needs a tolerant ANSWER regex.
- runner hangs: two occurrences (0 conns, 0% cpu, no flush) hours into long sessions; restart + resume recovers. Full run needs babysitting or a watchdog.
- self-pair position bias (share picking transcript 1): judge-specific and both directions - qwen3-8b .31-.43, qwen3-32b .33-.39, ling .54-.64, deepseek/glm ~neutral. Both-orders counterbalancing is doing real work.
- big-gap probe (epoch transcripts, gaps 24-37, weak judges): NOT monotone in gap. Pairs without gpt-5.5-low: .95-.98. Pairs with gpt-5.5-low (low effort, terse finals): .63-.78 despite LARGER gaps. Transcript style/verbosity dominates gap size - matches the AIME flat-curve finding.

## 2026-08-09/10 orientation correction + closed-member expansion (all verified)
- CRITICAL ANALYSIS FIX: runner score = P(picked first-named member), and pre-fix comparisons.csv named the weaker member first in 241/369 rows (deepseek pairs 65/73). Every pre-09 'judge accuracy' readout was contaminated; the apparent gpqa gap-curve (0.39->0.67) was the artifact - oriented truth declines (0.61->0.33). Fixed: generator emits stronger-first (no re-judging, logs are orientation-complete), viewer exporter + figures orient by ECI on load.
- corrected finding: judges competently rate the VISIBLE artifact, which anti-correlates with capability when the stronger model is a terse reasoning model. Verbose-stronger pairs: 0.65->0.76->0.98 with gap. Terse-stronger: chance or below.
- cheap-7 gpqa judging COMPLETE: 223 comparisons (7 judges x 9 members incl sonnet/o4-mini/gpt-4.1), 446 success phase-runs, ~66k verdicts, 0.1% unparsed, pins clean.
- gpt-4.1 aime test (verbose ECI-stronger member): judges pick it BELOW chance (0.17-0.44) even length-matched vs qwen3-8b; explanations show judges correctly penalizing its visible algebra slips - on aime the visible artifact anti-tracks capability. gpt-4.1 aime collection added (acc .317/.350, OpenAI pin).
- reasoning ablation (include_reasoning=true, cheap-6, both aime years) running under logs/data-expansion-reasoning-ablation/.
