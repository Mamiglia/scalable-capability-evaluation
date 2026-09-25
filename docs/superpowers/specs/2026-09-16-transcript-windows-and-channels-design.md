# Where in the transcript is the capability signal? Design

Research question 2 of the paper: the pairwise judge extracts a capability signal from
saturated-benchmark transcripts (question 1). This collection asks where that signal sits, in
two senses, by judging the same frozen train items under partial views of each transcript.

## Scope

- Pairwise judging only (direct judging scores at the shuffle null, so it cannot localise anything).
- Coding and cyber, semester 2026-I, judge Kimi-K2-Thinking with subject reasoning shown: the
  cells whose full-transcript baselines are `logs/judge/<domain>/2026-I/kimi-k2-thinking/24d6cc32`.
  Only these domains have tool calls. Other semesters and domains later, if the signal localises.
- Same frozen `2026-I_train` items, same pairs, same prompt and orders as the baseline.

## Position windows

A window is a fraction of the rendered transcript by character count, applied per transcript
before the existing character limit. `keep-k/n` shows the k-th n-th of the text and nothing else;
`drop-k/n` shows everything except that slice, with the existing `...` separator marking the gap.
Proposed launch set: keep each decile (10 runs), keep each quartile (4), drop each quartile (4).
Dropping a decile removes too little to move a judge and costs 90% of a full run each, so
drop-deciles are supported by the config but not launched.

Windows cut mid-line, as head/tail truncation already does; a role prefix or reasoning marker may
be split. Transcripts whose window leaves fewer than the existing 20-character minimum are skipped
by the existing filter, so short transcripts can drop out of keep-decile runs; every run reports its
retained pair count against the baseline.

Fractions, not fixed character counts, so transcript length is visible in every window. The
length-only ridge condition from the results analysis is what separates "signal in the text of
quarter 2" from "signal in the transcript's length".

## Channels

Three channels of the rendered transcript: `reasoning` (the `<THOUGHT_START>` blocks),
`text` (assistant prose, including media placeholders) and `tools` (a tool call together with its
tool output; an output without its call is unreadable). `only-<channel>` shows one channel;
`drop-<channel>` hides one. Six runs per domain. Turns left empty by the filter are not shown at
all, so a tools-only transcript is a list of calls and outputs, not a list of empty assistant lines.
The full rendering is byte-identical to today's when no channel spec is set.

`drop-reasoning` renders the same text as the existing `include_reasoning: false` twin configs;
it is collected under the channel label anyway, since no reasoning-hidden run exists for these
cells and the label keeps the six channel runs side by side.

## Mechanics

Two new `JudgeConfig` fields, `window` and `channels` (empty by default), enter the condition
hash only when set, so every existing hash and log directory is unchanged. The readable log
directory suffix generalises from `truncated-<cut>-<chars>` to a label built from whichever of
truncation, window and channels are set: `logs/judge_window-keep-2of10/...`,
`logs/judge_channels-only-tools/...`. Collection uses the stock command:

    uv run evaluate configs/domain/<domain>.yml --protocol chars --judge configs/judges/<set>/<file>.yml --semester 2026-I

which collects the missing verdicts and scores the run against the frozen theta*.

## Cost

Each condition re-judges every baseline pair: 13,744 verdicts for cyber 2026-I (coding is of the
same order). 24 conditions per domain is 48 runs and roughly 16 full-run equivalents of input
tokens, but the judge's own reasoning output per verdict dominates when inputs are short, so the
bill is closer to 48 baseline runs. The twelve pairwise truncation runs exhausted the OpenRouter
weekly key limit; this set is four times that. Launch order: channels (6 per domain), then
quartiles (8), then deciles (10), each batch stopping for a look at the scores before the next.

## Out of scope

Analysis figures for these runs (a loader like the one for truncated runs follows once the
verdicts exist). Direct judging. Math and science. Windows by message or turn boundaries.
