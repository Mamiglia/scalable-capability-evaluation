# collection run log

## 2026-08-08 smoke (limit 5)
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
