# Instructions for coding agents (Claude, Codex, ...)

Project overview and methodology: `README.md`. Data contracts: `data/CONTRACT.md`, `logs/CONTRACT.md`.

## Rules

- **Every judging protocol uses the frozen stratified train set, at most 100 items per task.** The
  reference freeze (`uv run reference`, `src/reference/freeze.py`) selects 25 EASY items per
  difficulty quartile, item-level, identical for every model, and records them as the
  `<semester>_train` column of `data/meta/<domain>/items.csv`. Pairwise, direct, feature-based or
  any new protocol judges only those items; never all EASY items, never a widened or re-drawn set.
  A challenge condition (truncation, hidden reasoning, ...) keeps exactly the retained items of the
  full campaign it is compared against.
- Harness code for campaigns and challenges lives under `src/challenges/<name>/`; `artifacts/<campaign>/`
  holds only frozen inputs, raw logs, results and audits.
- Do not launch paid collection on a handoff alone: propose the condition set (limits, judges,
  domains) and wait for the decision.
