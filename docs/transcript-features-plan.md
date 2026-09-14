# Transcript-feature GLM plan

Predict each model's domain-specific HARD-item capability (θ*) from its correct, first-attempt transcripts on the current task's selected EASY items. No model metadata.

## Three variants

1. `glm_length`: 3 length/visibility features.
2. `glm_regex`: those 3 + 9 regex rates = 12 features.
3. `glm_structure`: those 12 + tool-call count + action share + 6 transitions + 4 availability flags = 24 features.

See the [feature table](transcript-feature-table.md) for definitions, extraction rules, and suggested changes.

## Features

- **Length:** `log(1 + characters)` for assistant text (including visible reasoning) and reasoning separately; a visible-reasoning indicator.
- **Regex:** pronouns, hedging, uncertainty, confidence, backtracking, verification, causal connectives, contrast, and question marks. Reuse the `judge-dissect` lexicon; express counts per 1,000 prose words.
- **Structure:** log total tool calls, action calls / total calls, six conditional RST/tool transition rates, and four source-availability flags. Count calls from structured records; action share uses classified call nodes. No full transition matrix.

Count linguistic features in assistant prose and available reasoning, excluding prompts, tool outputs, code, and tool-call payloads. Extract tool calls from structured messages.

## Fit and evaluation

Calculate rates per transcript, then average features across each model's eligible transcripts: one regression row per model. Fit a Gaussian GLM with identity link, standardisation, and ridge regularisation. Learn preprocessing and coefficients only from references in **all previous rounds**. No item centring or hierarchical fit initially.

Add missing historical reference θ* labels on the current HARD set, preserving existing splits and judging tasks. Report missing coverage explicitly; absent transcripts are not zero-valued observations.

## Implementation

1. Add `src/protocols/transcript_features/`, reusing `TabularRegression` and the existing evaluator.
2. Wire structured messages, RST/tool labels, and all-history reference selection.
3. Check extraction, transition denominators, aggregation, and reference-only fitting; compare the three variants using target MSE and coverage.
