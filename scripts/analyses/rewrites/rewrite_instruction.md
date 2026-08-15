# Paraphrase rewrite instruction

Sent verbatim to both rewriters. Everything below the rule is the prompt; edit freely.

Notes on why clauses are here, so edits do not silently undo a fix:

- The previous instruction preserved content but flattened structure: a 915-word chemistry
  transcript with 38 numbered steps came back as reordered prose, and an in-flight
  `"35) Wait! Maybe the question meant cyclohexyl chloride..."` became a settled
  `"The problem structure may have a typo"`. Judges respond to backtracking (-0.19 log-odds
  per SD), so flattening self-correction removes reasoning-process evidence, not just style.
- The previous instruction said the paraphrase "must not be longer than the original", which
  both rewriters read as a compression target: compression ran 0.27-0.85 by model, the
  within-pair length gap fell from a median of 120 words to 51, and which member was longer
  flipped in 22.9% of cells. Length is manipulated separately by the length-matching
  conditions, so this instruction should hold length roughly constant instead.
- At high compression the rewriters stopped paraphrasing and began repairing broken
  reasoning, which inverts what the condition measures. Preserving step order and dead ends
  is what guards against that.

---

Rewrite the following answer transcript in your voice.

Preserve the reasoning exactly. Every step in the same order, including false starts, dead
ends, self-corrections and moments of uncertainty. If the original hesitates, doubts itself,
or changes its mind, the rewrite must do so at the same point.

Do not fix, complete, improve, reorder or tidy the reasoning, and do not add anything. Do not
convert a step-by-step derivation into summary prose — if the original works through numbered
steps, the rewrite works through the same steps.

Change only wording and register: replace idiosyncratic phrasing, tone and formatting habits
with plain neutral prose of a consistent register.

Keep the final answer line exactly as it is.
