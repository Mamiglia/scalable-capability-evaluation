# Predicting further ahead: why does error grow with older references?

2026-09-23. Science only. Core task design throughout, with only the reference semester changed:
frozen stratified train set (40 of the frozen 100 EASY items), reasoning shown, and the judge
chosen by leave-one-out error from the three strongest references.

## Question

Every current task predicts one semester ahead. If error grows when the references (and so the
judge) come from further back, there are three candidate causes:

1. the targets are too different from the references (capability distance, or the reasoning shift);
2. the judge is too weak;
3. the references are too weak to anchor the targets.

The judge is one of the references, so causes 2 and 3 move together within the Pairwise judge.
The design separates cause 1 from causes 2 and 3, and uses the protocols without a judge to split
2 from 3.

## Design

Each new task is a (reference semester → target semester) pair.

- Targets fixed at 2026-I, references moved back: 2025-I→2026-I, 2024-II→2026-I. Distance grows
  and the references and judge get weaker together.
- References fixed at 2024-II, targets moved forward: 2024-II→2025-II (2024-II→2025-I exists,
  2024-II→2026-I is shared with the series above). Distance grows with references and judge
  fixed, so only cause 1 changes.
- References fixed at 2025-I, older targets: 2025-I→2024-II, 2025-I→2024-I. The same distances
  with strong references and judge. If error depends only on distance these match the forward
  tasks; if they are clearly better, the forward tasks suffer from weak references or judge.
  2025-I→2024-II is the exact mirror of the existing 2024-II→2025-I: the same sixteen models.
- Free, from existing tasks: every one-semester task from 2024-I to 2026-I varies reference and
  judge strength at a fixed one-semester distance.

New tasks (5): 2025-I→2026-I, 2024-II→2026-I, 2024-II→2025-II, 2025-I→2024-II, 2025-I→2024-I.
2024-I references are left out: their judge is GPT-4 Turbo, which would be most of the cost.

Train items and judge come from the existing task with the same references. The EASY/HARD split
is set by the three strongest references and the judge is selected on reference–reference
verdicts, so only the target–reference pairs are new. Judges (the paper's: lowest leave-one-out
error, linear variant, core condition): 2025-I references → MiniMax-M1, 2024-II references →
Qwen2.5-72B.

Qwen2.5-72B has a 32k context, a limitation of the protocol with this judge. In the existing
Science 2026-I reasoning-shown runs, 4.8% of judgments had more than 30k input tokens (median
2.9k, max 112k). For the two Qwen2.5-72B tasks each transcript is cut to its first 25,000
characters (`max_length=25000`, `truncate=head`). Measured with the Qwen2.5 tokenizer on existing
long Science 2026-I judgments: median 2.9 characters per token, 5th percentile 2.6, minimum 1.5.
Two transcripts at 25k characters are ~19.5k tokens at the 5th percentile, well inside 32k with
the prompt, question and output; only the densest few could still overflow. Only transcripts
longer than 25k characters are cut. MiniMax-M1 (1M context) is not
truncated.

Older targets on the 2025-I references' items: they solve a median of 78 of the 100 train items
(minimum 58), so ~31 (minimum ~23) of the 40 are judged, and θ* rests on ~95 HARD items (299 for
the normal forward task).

## Analysis

Per protocol (Pairwise judge, Embedding, Deterministic features), per-target squared error against:

- distance: target θ* minus the strongest reference's θ* (negative for older targets);
- reference strength: the strongest reference's θ*;
- a reasoning-target flag.

Reading the results:

- Error grows with distance when references are fixed → cause 1.
- Forward tasks are worse than older-target tasks at matched distance → causes 2 or 3.
- Embedding and Deterministic features (no judge) degrade as much as the Pairwise judge when
  references move back → cause 3. Pairwise degrades more → cause 2.
- Paired t-tests on squared error over targets within each comparison, as in the paper's
  within-protocol sections. With 8 targets per task, expect effect sizes, not per-task
  significance.

## Code

All work on a new git worktree, so none of it touches the main analysis.

`src/reference/freeze.py` hard-codes the previous semester as the references (`roster`) and keys
the train column by the target semester alone. It needs a reference semester argument and a
column key naming both semesters. Local, no cost.

## Cost

Per task, 64 target–reference pairs × 40 items × 2 orders × the both-correct fraction (~0.9
forward, ~0.8 for older targets). Embedding and Deterministic features cost nothing.

Input tokens per judgment from the existing reasoning-shown core runs with the same target
semester (2026-I 8.2k, 2025-II 3k; older targets with 2025-I references ~2k), upper bounds since
those runs paired targets with newer, longer reference transcripts. Output: MiniMax-M1 ~1k,
Qwen2.5-72B ~70. OpenRouter prices of the pinned endpoints (per M tokens, in/out): MiniMax-M1
(novita) $0.55/$2.20, Qwen2.5-72B (deepinfra) $0.36/$0.40.

| Task | Judge | Upper-bound cost |
|---|---|---|
| 2025-I→2026-I | MiniMax-M1 | $31 |
| 2024-II→2026-I | Qwen2.5-72B | $14 |
| 2024-II→2025-II | Qwen2.5-72B | $5 |
| 2025-I→2024-II | MiniMax-M1 | $11 |
| 2025-I→2024-I | MiniMax-M1 | $11 |

Total ~$72 (100 items: ~$180).
