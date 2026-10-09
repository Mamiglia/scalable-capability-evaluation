# Instructions for coding agents (Claude, Codex, ...)

Project overview and methodology: `README.md`. Data contracts: `data/CONTRACT.md`, `logs/CONTRACT.md`.

## Data location

All project data lives in the Hugging Face org **generality-labs** (https://huggingface.co/generality-labs),
migrated from LASR-G5 in September 2026: `generality-labs/benchmarks` (data/), `generality-labs/judging`
(logs/), `generality-labs/glm_features` (glm_features/). Any `LASR-G5/...` URL in code or docs is stale;
use the generality-labs one.

## Rules

- **Every judging protocol uses the frozen stratified train set, at most 100 items per task.** The
  reference freeze (`uv run reference`, `src/reference/freeze.py`) selects 25 EASY items per
  difficulty quartile, item-level, identical for every model, and records them as the
  `train` column of `data/meta/task_items.csv`. Pairwise, direct, feature-based or
  any new protocol judges only those items; never all EASY items, never a widened or re-drawn set.
  A challenge condition (truncation, hidden reasoning, ...) keeps exactly the retained items of the
  full campaign it is compared against.
- Harness code for campaigns and challenges lives under `src/challenges/<name>/`; `artifacts/<campaign>/`
  holds only frozen inputs, raw logs, results and audits.
- Do not launch paid collection on a handoff alone: propose the condition set (limits, judges,
  domains) and wait for the decision.

## Figure colour palette

Protocols, with the names the paper uses (use these names in figures, tables and prose; the
code keys are in brackets). Length alone (`glm_length`) is a diagnostic, not a paper protocol:
leave it out of paper figures.

- Deterministic features (`glm_regex`): `#eb6834`
- Embedding (`embedding`): `#4a3aa7`
- Absolute scoring (`direct`): `#1baf7a`
- Pairwise judge (`pairwise`): `#2a78d6`
- Length (`glm_length`, diagnostic only): `#f1a48a`

Shades of Pairwise judge (ex): `#6aa1e2`, `#2260ab`
Triadic combinations for Pairwise judge: `#d62a78`, `#78d62a`
Extra, for a tenuous colour: `#f8cbbc`

Domains (marker shape): SCIENCE circle, MATH square, CODING triangle, CYBER diamond.
