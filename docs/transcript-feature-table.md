# Transcript GLM feature table

The three variants now use **3, 12, and 24 predictors**. The full transition matrix has been replaced by an explicit shortlist; there are no extra diagnostic columns outside these counts.

| Variant | Included features | Count |
| --- | --- | ---: |
| `glm_length` | Length/visibility | 3 |
| `glm_regex` | Length/visibility + lexical rates | 12 |
| `glm_structure` | Above + tool-call count + action share + six transition rates + four availability flags | 24 |

The six transitions below are a **starting shortlist for review**, not features established as predictive by the earlier pilot. The feature calculations are implemented; connecting the production RST/tool labeller is still pending.

## Shared extraction rules

Use correct first-attempt transcripts on the task's selected EASY items. Calculate each feature per transcript, then average equally across a model's available items. Fit standardisation and the GLM on earlier-round models only.

**Length text:** assistant text and visible raw reasoning, joined with newlines. Includes code, maths and the final answer; excludes prompts, tool-call arguments, tool outputs and rendering markers.

**Lexical text:** the same assistant text, with code, recognised maths spans and `ANSWER:` lines removed. Rates are matches per 1,000 whitespace-separated words. Match whole words/phrases, longest first within each category; normalise curly apostrophes. An observed empty prose body gives zero rates; a missing transcript remains missing.

**Structure:** chronological RST/protected-span/tool labels. `organization` and `explanation` are parser relations, not guarantees of planning or sound reasoning. Tool results remain separate nodes. For a selected A → B rate, divide its count by **all outgoing transitions from A**, including unselected destinations. The final node has no outgoing transition.

**Tool use:** count every structured assistant tool-call entry, including repeated calls. Action share uses the classified `tool_act` / `tool_inspect` nodes, including a final call. With at least one call, inspection share is 1 minus action share, so a second share is unnecessary. No calls gives action share zero; the zero call count identifies this case. Missing message history gives NaN for both.

## Length and regex: 12 features

| Feature | Definition | How extracted | Description / interpretation |
| --- | --- | --- | --- |
| `log_chars` | ln(1 + assistant-text characters) | Count the length text described above. | Overall visible verbosity. |
| `log_reasoning_chars` | ln(1 + visible-reasoning characters) | Count supported, non-redacted reasoning blocks only. | Visible reasoning volume; zero can mean unavailable reasoning. |
| `visible_reasoning` | 1 if visible reasoning exists; otherwise 0 | Check whether the extracted reasoning string is nonempty. | At model level, the fraction of items exposing reasoning. |
| `personal_pronouns` | First-person pronoun rate | Terms such as “I”, “we”, “my”, “our”. Special handling prevents common mathematical uses of “I” from matching. | Self-reference and narration. |
| `hedging` | Hedging phrase rate | “perhaps”, “might”, “likely”, “I think”, etc. | Tentative phrasing. |
| `backtracking` | Reconsideration cue rate | “oh wait”, “reconsider”, “start over”, “hm/hmm/hmmm”, etc. Overlapping phrases count once. | Verbal reconsideration, not verified self-correction. |
| `uncertainty` | Explicit uncertainty rate | “unsure”, “not sure”, “I don't know”, etc. | Stated uncertainty. |
| `confidence` | Strong-confidence phrase rate | “definitely”, “obviously”, “must be”, etc. | Assertive wording, not calibrated confidence. |
| `questions` | Question-mark rate | Count `?` in lexical text. | Question-like phrasing, including self-questioning. |
| `verification` | Verification cue rate | “check”, “verify”, “confirm”, “substitute back”, etc. | Mentions of checking; does not establish that a check succeeded. |
| `connectives` | Causal/consequential connective rate | “because”, “therefore”, “hence”, “given that”, etc. | Explicit links between statements. |
| `contrast` | Contrast/alternative marker rate | “however”, “but”, “instead”, “alternatively”, etc. | Contrasting statements or considering alternatives. |

The exact word lists are in [TERMS](../src/protocols/transcript_features/features.py); examples above are not exhaustive. Categories may overlap: “I think” contributes to pronouns and hedging.

## Structure: 12 additional features

Rate columns are named `transition__<source>__<destination>`. Only adjacent nodes count; graph edges and two-step transitions are not used.

| Feature | Definition | How extracted | Description / interpretation |
| --- | --- | --- | --- |
| `log_tool_calls` | ln(1 + total tool calls) | Count entries in assistant messages' structured `tool_calls` lists. Include repeats; do not count mentions in text. | External work beyond visible text length. |
| `tool_action_share` | Action calls / (action + inspection calls) | Count `tool_act` and `tool_inspect` nodes; use every call, including the final node. | Balance of acting versus gathering evidence. |
| `organization` → `equation` | Fraction of outgoing organization nodes followed by an equation | Count adjacent labelled pairs; divide by outgoing organization nodes. | Organising prose leading into formal working. |
| `explanation` → `equation` | Fraction of outgoing explanation nodes followed by an equation | Count pairs; divide by outgoing explanation nodes. | Explanatory prose leading into formal working. |
| `equation` → `equation` | Fraction of outgoing equation nodes followed by another equation | Count pairs; divide by outgoing equation nodes. | Consecutive formal steps. |
| `equation` → `explanation` | Fraction of outgoing equation nodes followed by an explanation | Count pairs; divide by outgoing equation nodes. | An explanation following formal working. |
| `tool_output` → `tool_inspect` | Fraction of outgoing tool outputs followed immediately by an inspection call | Count pairs; divide by outgoing tool-output nodes. | Further information gathering without intervening prose. |
| `tool_output` → `tool_act` | Fraction of outgoing tool outputs followed immediately by an action call | Count pairs; divide by outgoing tool-output nodes. | Acting immediately after a result, without intervening prose. |
| `source_observed__organization` | 1 if an organization node has an outgoing transition | Check the denominator of organization-source rates. | Marks whether these rates had an opportunity to be observed. |
| `source_observed__explanation` | 1 if an explanation node has an outgoing transition | Check the denominator of explanation-source rates. | Same, for explanation. |
| `source_observed__equation` | 1 if an equation node has an outgoing transition | Check the denominator of equation-source rates. | Same, for equations. |
| `source_observed__tool_output` | 1 if a tool-output node has an outgoing transition | Check the denominator of tool-output-source rates. | Same, for tool outputs. |

If a source has no outgoing transition, its rates and availability flag are zero. After averaging over items, the flag is the fraction of items with an observed source. The rate is currently a **zero-filled per-item mean**, not a pooled conditional probability or an average restricted to source-present items. Missing message history leaves structure features NaN.

## Removals applied

| Removed feature(s) | Previous definition/extraction | Why removed |
| --- | --- | --- |
| 619 unselected transition rates | Every ordered pair among 25 labels | The full matrix expanded the GLM to hundreds of mostly sparse predictors. The two provisional organization/tool transitions were replaced by call count and action share. |
| 21 unused source flags | An outgoing-transition indicator for every label | Only the four sources used by the shortlist need flags. |
| `log_assistant_turns` | ln(1 + number of assistant messages) | Unused diagnostic, never selected by these GLMs. |
| `log_distinct_tools` | ln(1 + number of distinct tool function names) | Unused diagnostic; also strongly tied to the available tool set. |
| `log_repeated_tool_calls` | ln(1 + repeated function/JSON-argument pairs beyond their first occurrence) | Unused diagnostic; a repeat rate would be a more natural candidate if needed. |

## Suggested changes for review — not applied

Treat 24 as a ceiling: additions should replace a weaker feature, rather than automatically expand the set. Check support/correlation on reference models only, before inspecting target forecast errors.

| Suggestion | Definition / extraction | Reason |
| --- | --- | --- |
| Add code share and/or maths share | Recognised code/maths characters divided by assistant-text characters. | Readable measures from the older pilot; could replace sparse equation-specific transitions, especially in coding. |
| Add repeated-call rate | Repeated identical function/argument pairs divided by total calls. | Captures repeated actions with less dependence on trajectory length than a raw count. |
| Add checking/backtracking after results | Fraction of tool outputs whose next assistant prose unit contains the existing verification/backtracking cues. | Retains sequence information from the pilot's motif ideas; requires clause text as well as labels. |
| Consider broader prose destinations | For example, equation → any prose, grouping RST prose labels only for that feature. | Strict equation → explanation can miss explanatory text labelled `context` or `mode`. |
| Ablate the visibility indicator | Omit `visible_reasoning` while keeping the other columns fixed. | It partly measures provider disclosure. This matters when comparing open and closed models. |
| Compare supported-item averaging | Average each conditional rate only over items with an outgoing source; retain availability separately. | Separates transition behaviour from source frequency more cleanly than the current zero-filled mean. |

No predictive benefit is claimed for the shortlist yet. The older [lean configuration](https://github.com/Mamiglia/small-judge/blob/697f1e725c328b5b170fdd23dca01da47beb5767/scripts/verdict_dissection/glm/predictors-lean.yaml) had 29 features, including other feature families; the mathematical/tool themes above also draw on its [named motif features](https://github.com/Mamiglia/small-judge/blob/697f1e725c328b5b170fdd23dca01da47beb5767/scripts/verdict_dissection/features_graph_motifs.py), with adjacency explicitly replacing graph-child relations.
