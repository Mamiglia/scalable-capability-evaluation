---
name: audit-evals
description: Sweep the collected .eval logs in data/ for defects that would corrupt the weak-to-strong judging study - scorer/parser failures, reasoning that is missing, unrecovered, or a provider summary rather than raw CoT, truncated or empty transcripts, media the judge cannot see, self-identifying models, degenerate output, accuracy outliers. Also covers stamping edit provenance into any log an audit fixes. Use when the user asks to audit, validate, sanity-check, or "go through" the eval logs / a benchmark's data, when a model's accuracy looks implausible, or before a collection sweep is declared final.
---

# Auditing the collected eval logs

`data/` holds ~4700 `.eval` logs across ~20 benchmarks (25 GB). You cannot read them all
by hand, and you must not try. The job is: **run the three scripted sweeps, then hand-triage
only what they flag.** Coverage comes from the scripts; judgement comes from you.

Read `data/CONTRACT.md` (the binding rules for `data/`: layout, repeats, log validity,
edit stamping, sync), then `CLAUDE.md` (research framing) and `AGENTS.md` (§Known failure
modes) if they are not already in context. What this skill finds is measured against that
contract; `bash scripts/audit/logs/check.sh` is its structural gate. The `reading-logs` and `analyzing-logs` skills
cover the Inspect log APIs — use them for triage, and never unzip a `.eval`.

## Why these defects matter here

The study filters to questions **both models answered correctly** (`require_both_correct`),
then asks a judge to pick the more capable model from the two transcripts. So:

- A **scorer that under-parses** removes true-correct items from the pool and biases which
  model looks weak. This has already happened three times (see Precedents).
- **Missing reasoning** where the run asked for it makes the `include_reasoning` control
  meaningless for that model — the on/off conditions become identical.
- **Reasoning that is a *summary*** rather than raw CoT breaks the comparison the study
  rests on: see the section below, this is the subtlest defect in the data.
- **Truncation, media placeholders, empty transcripts** give the judge something other than
  capability to read, or nothing at all.
- **Self-identification** ("I'm Claude, made by Anthropic") lets the judge recognise the model
  instead of assessing it — a direct confound with the ground-truth capability index.
- **Degenerate repetition** correlates with length, which `truncate_frac` is supposed to control.

## Procedure

### 1. Scope

Default to all of `data/`. Skip `data/legacy/` (archived, excluded from analysis). Narrow to
`data/<bench>` when the user names one. Note up front how many logs are in scope.

### 2. Structure sweep (already-written, cheap)

```bash
uv run python scripts/audit/logs/validate_data.py            # repeats, completeness, replayed repeats
uv run python scripts/audit/logs/report_collection.py data   # accuracy, cost, reasoning share per run
```

`validate_data.py` exits 1 on findings. **Never pass `--delete`** unless the user explicitly
asks — it removes logs.

### 3. Content sweep (the per-sample checks)

```bash
uv run python scripts/audit/logs/audit_evals.py                      # all of data/, 50 samples per log
uv run python scripts/audit/logs/audit_evals.py data/gpqa --samples 0   # exhaustive, one bench
uv run python scripts/audit/logs/audit_evals.py --report             # re-print without rescanning
```

Results are cached in `analysis/eval_audit.jsonl` keyed by path, so a second run only scans
new logs. Pass `--refresh` after a log is rescored in place. `--selfcheck` runs the flag-logic
asserts. The default `--samples 50` is a sampling ceiling *within* a log, not across logs:
every log is visited. Say so in the write-up, and re-run the flagged ones with `--samples 0`.

### 4. Triage each flag

The script reports *rates*, not verdicts. Confirm every flag against the transcripts before
calling it a defect — most flags have a benign reading.

| Flag | Benign reading | Real-defect reading | Confirm by |
| --- | --- | --- | --- |
| `unparsed` | weak old model genuinely emitted no answer | model wrote `ANSWER: $C$` / `\boxed{}` and the scorer's regex missed it | print `score.answer` next to `output.completion[-200:]` for a few samples |
| `acc_outlier` | the model is simply bad at this bench | accuracy near 0 while transcripts look competent | cross-check the model's public score (`benchmark-scores` skill); compare with `unparsed` |
| `truncated` | long-CoT model on a hard bench | systematic — `max_tokens` cap in the model config too low | check `generate_args.max_tokens` in `configs/models/**` vs the observed output tokens |
| `no_reasoning` | the run's `reasoning.enabled` never applied by that provider | broken provider pin — transcripts unusable for the reasoning condition | `scripts/audit_configs.py`; AGENTS.md §Provider pinning |
| `reas_summary` | — | not a defect in the data — the model's provider hides its CoT, so it drops out of the reasoning arm by design. Report it, don't fix it | print `.redacted`, `.reasoning[:120]`, `.summary[:120]` for one block |
| `reas_thin` | short CoT on an easy item | a summary passed off as raw CoT | read one block (third-person prose = summary); if confirmed, add the model to `normalize_reasoning.py --summary-models` |
| `reas_lost` | — | always real: billed for reasoning, none recovered | check the provider pin |
| `edited` | a fix already applied and recorded | — | read `eval.metadata["edits"]`; not a defect, but the log is not as-collected |
| `think_open` | rare budget exhaustion | frequent — the model spends its whole budget on CoT and never answers | `strip_think` drops everything, so the judge sees an empty answer |
| `media` | one image-bearing item | a bench where many items are non-text | the judge only ever sees `[Image]`; consider filtering the item out |
| `self_id` | the question is *about* a vendor | assistant text names its own model | read the matching span; this is the strongest confound, escalate it |
| `degenerate` | structured/tabular output | repetition loop | read the transcript tail |
| `no_output` / `empty_tx` / `render_err` | — | always real | `render_err` = a content block `judge/util.py` does not handle |
| `multi_scorer` | — | always real | `extract_score` raises on >1 scorer; the bench config must expose one |

Escalate a flag to a finding only when you have seen the evidence. Prefer 2-3 quoted samples
over prose. When several logs share one cause (a model, a provider pin, a bench's scorer),
report the cause once, not once per log.

### 5. Report

Group findings by cause, most consequential first. For each: what it is, how many logs and
which, the evidence, the effect on the study (does it bias pairing? break a control? just
lose items?), and the cheapest fix. Distinguish **new** findings from ones already recorded
in `data/collection_runs.md`.

## Reasoning representation (check this on every model)

Only raw chain-of-thought is admissible evidence for the judge. Everything else a provider
returns in a reasoning slot — a summary, an encrypted blob — is not comparable to a CoT, and
judging one against the other measures the provider's disclosure policy, not the model.

`ContentReasoning` has `.reasoning` (the CoT), `.summary` (a provider paraphrase), and
`.redacted`. Its `.text` **property** silently swaps in the summary when `redacted` is set —
`src/judge/util.py` deliberately does not use it.

### What reaches the judge

| Stored as | Who | Rendered as |
| --- | --- | --- |
| `.reasoning`, `redacted=False` | deepseek, qwen, glm, kimi, nemotron | `<THOUGHT_START>…<THOUGHT_END>` — **preferred over everything** |
| `.summary` on a redacted block | o3, gpt-5 | **nothing** |
| encrypted blob in `.reasoning` on a redacted block (`gAAAAAB…`) | o3, gpt-5 | **nothing, ever** |
| inline `<think>…</think>` in a text block | deepseek via plain-text transport | normalized into a reasoning block (below) |
| thought summary in `.reasoning`, `redacted=False` | gemini | normalized into `.summary` (below) |

The rule in `util.format_reasoning` is one line: show `.reasoning` when the block is not
redacted, never anything else.

### Normalizing the logs, not the renderer

The last two rows are **log defects**, fixed once at the source rather than parsed on every
render — so that every consumer (judge, `report_collection.py`, `audit_evals.py`) sees one
shape instead of each special-casing the same quirk:

```bash
uv run python scripts/audit/logs/normalize_reasoning.py data/            # dry run, reports only
uv run python scripts/audit/logs/normalize_reasoning.py data/ --write    # rewrites + stamps provenance
```

- **inline `<think>`** → split into `ContentReasoning(reasoning=…)` + the answer text. Same
  CoT, same rendering as a native block, so the transport is invisible to the judge.
- **gemini thought summaries** → moved from `.reasoning` to `.summary` with `redacted=True`,
  which is what they actually are: the real CoT never left the provider. Nothing in the log
  marks this, so it is gated on an explicit `--summary-models` list and **never guessed** —
  confirm with the token ratio first (below), then add the model to the list.

### Consequence: pairs drop out of the reasoning arm

A model whose provider discloses no CoT reads as having **no reasoning**, so
`build_dataset`'s one-side-only guard refuses to pair it against a model that does. Verified
on HLE:

| Pair | `include_reasoning: true` | `false` |
| --- | --- | --- |
| o3 vs deepseek-v3.2 | **0** (refused) | 446 |
| gpt-5 vs glm-4.6 | **0** (refused) | 628 |
| gemini-3.5-flash vs glm-4.6 | 834 → **0 after normalization** | 834 |
| deepseek-v3.2 vs glm-4.6 | 608 | 608 |

That is intended. Do not "fix" it by re-enabling summaries. The no-reasoning arm keeps every
pair, so it is the shape-immune comparison — anchor headline claims there.

### Finding the next one

A provider returning only a summary in `.reasoning` with `redacted=False` is indistinguishable
from raw CoT in the log. `audit_evals.py` flags it as `reas_thin` — recovered characters far
below the billed `reasoning_tokens`. Confirm by eye before adding a model to
`--summary-models`: a thought summary reads as tidy third-person prose, raw CoT reads as
first-person and meandering. Report per model: shape, mean recovered chars, mean billed
`reasoning_tokens`.

## Recording an edit — mandatory

Any `.eval` an audit changes (rescoring, re-grading, patching) **must carry the record inside
the file**. `data/collection_runs.md` and git history live in a different repo and detach from
the log the moment it is copied.

```python
from eval_edits import stamp_edit          # scripts/audit/logs/eval_edits.py
stamp_edit(log, tool="scripts/judge/rescore_mcq.py", reason="stock parser missed `ANSWER: $C$`",
           source=input_path, samples_changed=185,
           before={"accuracy": 0.061}, after={"accuracy": 0.778})
write_eval_log(log, out)
```

It appends to `eval.metadata["edits"]` (append-only, never rewrite a past record) and tags the
log `edited` + `edited:<tool>`, so a header-only read sees it and `audit_evals.py` reports it.
`rescore_mcq.py` already does this. Any new fixer script must call it too — an edit with no
record in the file is the defect. Still log the run in `data/collection_runs.md` as well; the
two serve different readers.

Logs rescored before this existed (muse-glimmer, nemotron-3.5-lightning, gpqa 2026-08-25)
carry no record — they are identifiable only from `collection_runs.md`. Note that when auditing
them; do not assume an unstamped log is as-collected.

## Fixes — propose, do not apply

- **Parser too strict** → `scripts/judge/rescore_mcq.py` rescores with a tolerant `ANSWER`
  regex, stamps provenance, and refuses to overwrite its input. Check first that no sample has
  more than one `ANSWER:` match, or the tolerant regex can pick the wrong one. Confirm it
  actually helps before proposing it: `yi-34b-chat` flags 40% unparsed but rescoring moves it
  only 0.157→0.167, because that model ignores the answer format entirely — a different defect.
- **Bad provider pin / token cap** → fix `configs/models/**`, then re-collect. Re-collection
  costs money: say how much before proposing it (`report_collection.py` prints per-run cost).
- **Confounded items** → filter at the domain level (`configs/domain/*.yml`), not by editing logs.

**Never mutate `data/`**: it is its own git repo synced to Hugging Face. No `git` commands
there, no deletions, no in-place edits, unless the user explicitly asks. Anything the user
does accept belongs in `data/collection_runs.md`.

### If the user does accept a fix that rewrites logs

`data/` is a **git submodule** of the parent repo, tracking
`https://huggingface.co/datasets/LASR-G5/benchmarks` with git-LFS on `*.eval`, `*.json`,
`*.eval-set-id`. The parent pins an exact commit, so a rewrite is not published until both
halves move, **in this order**:

```bash
cd data && git add <the rewritten logs> && git commit -m "..." && git push origin main
cd .. && git add data && git commit -m "data: bump pin to <sha> (<what changed>)"
```

The pin moves *after* the push, never before — otherwise it names a commit nobody can fetch.

- **git is the only transport.** `hf upload` / `huggingface_hub` write commits server-side that
  the local checkout does not have; the pin then dangles and the next `git push` is rejected or
  conflicts on LFS pointer files. If a push is rejected as behind, someone used the other
  transport: `git fetch origin main && git rebase origin/main`, then push.
- **Credentials.** git-lfs needs a credential for `huggingface.co` on *fetch and checkout*, not
  just push — a rebase that pulls new `.eval` objects fails with `smudge filter lfs failed` /
  "Git credentials … not found" if none is configured. Either persist one
  (`git config --global credential.helper store` then
  `hf auth login --token $HF_TOKEN --add-to-git-credential`) or pass it per command:
  `git -c 'credential.helper=!f(){ echo username=hf; echo "password=$HF_TOKEN"; };f' <cmd>`
  after sourcing `.env`.
- `data/legacy/` (superseded runs) and `data/meta/` (domain artifacts, regenerated by
  `uv run domain`) are gitignored inside `data/` and stay local — never push them.
- `logs/` (judge eval-sets) is gitignored in the parent but is **its own git repo** synced to
  `https://huggingface.co/datasets/LASR-G5/judging` — rules in `logs/CONTRACT.md` (the
  `judge-logs` skill covers handling). It is derived —
  temperature 0 with the generate cache on, reproducible from `data/` + `configs/experiments/`
  — so audit findings about judging must cite the config, not a log path someone else lacks.

## Precedents (what a real find looks like)

- **gemini-3.1-flash-lite / gpqa**: drifted to `ANSWER: $C$`; stock scorer read `""` →
  measured accuracy 0.004 vs a true 0.731.
- **muse-glimmer-30b, nemotron-3.5-lightning / gpqa**: same `$LETTER$` form; 0.646→0.798 and
  0.061→0.778 after `rescore_mcq.py`. Folded into the canonical dirs 2026-08-25. Note
  `logs.json` beside a rescored log still carries the pre-rescore accuracy — only the `.eval`
  is authoritative.
- **nemotron-3-super-120b / gpqa**: ~30% unparsed *and* ~35% truncated — still open at the
  time of writing. Two causes stacked in one log; do not stop at the first one you find.
- **o3, gpt-5 / hle**: `reas_summary 100%` — every reasoning block redacted. Their transcripts
  are now byte-identical with reasoning on and off, so they pair only in the no-reasoning arm.
- **gemini-3.5-flash**: `reas_thin 67%` — a thought summary with no redaction flag, so it
  rendered as CoT and still paired (834 samples against glm-4.6). Caught only by comparing
  recovered chars against billed reasoning tokens; fixed by `normalize_reasoning.py`.
