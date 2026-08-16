# Style-standardisation rewrite instruction — draft

This is a draft for review. It is not yet used by the rewrite pipeline.

Design intent:

- Treat style standardisation and transcript truncation as separate interventions.
- Do not impose a word-count target: forced length preservation can make the rewriter add filler.
- Require the minimum edits needed for style standardisation, without summarising or deleting
  reasoning. Any length change should arise naturally from those edits.
- Remove the three style-feature families motivated by the GLM: personal pronouns, hedging,
  and conversational backtracking.
- Preserve the reasoning, including substantive uncertainty, conditions, probability judgments,
  calculations, errors, abandoned approaches, and corrections.
- Preserve all answer-labelled lines and any content following them in their original order.

---

You are performing a controlled transformation for a research experiment.

Rewrite the supplied answer transcript so that it expresses the same reasoning in plain,
neutral, impersonal prose. Standardise only the targeted stylistic features. Do not solve the
problem yourself and do not improve the answer.

## Content fidelity

Preserve every substantive claim, assumption, condition, probability judgment, calculation,
equation, comparison, error, and conclusion. Preserve the order and logical dependencies of the
reasoning. Do not add facts, examples, justifications, or reasoning that are absent from the
original. Do not repair an error or make the argument more convincing.

Preserve any false start or abandoned approach that contains a distinct calculation, factual
claim, assumption, or candidate conclusion. Remove only empty verbal stalls and conversational
self-interruption markers. Express the transition to the corrected reasoning neutrally and at
the same point.

The following examples illustrate different acceptable constructions; they are not templates.
Vary the wording and sentence structure to fit the original transcript.

- “Wait, that sign is wrong” can become “The initial sign is incorrect; using the opposite sign
  gives ...”.
- “Actually, option B cannot work because the boundary condition fails” can become “Option B is
  initially considered, but the boundary condition rules it out.”
- “Maybe the reaction follows mechanism X. On second thought, the observed product contradicts
  that” can become “Mechanism X provides an initial candidate. Its predicted product conflicts
  with the observation, so the reasoning proceeds with the alternative mechanism.”
- A longer failed derivation can be followed by a transition such as “This approach is flawed
  because ...”, provided that the derivation and the reason it fails remain visible.

Remove speaker-centred or conversational hedging, but preserve substantive uncertainty,
conditionality, and probability judgments impersonally without strengthening or weakening them.
Constructions such as “Under the assumption that…” and “If X holds…” are allowed. For example:

- “I think X is likely” can become “The evidence favors X”.
- “X is most likely under assumption A” can become “Under assumption A, X has the greatest
  support”.
- “This may indicate X” can become “This observation is consistent with X but does not establish
  it”.
- “Possibly X or Y” can become “The available information does not distinguish X from Y”.

Retain equations, technical notation, numbered derivations, and step ordering where present. Do
not convert a detailed derivation into summary prose.

Make the minimum changes needed to standardise the targeted style features. Do not target a word
count. Do not add filler to preserve length, and do not shorten the transcript by summarising,
condensing, deleting reasoning, or removing repetition that is part of the original transcript.
Any change in length should be only the natural consequence of replacing the targeted stylistic
language.

## Targeted style standardisation

Use plain, neutral, impersonal, declarative prose.

Do not use first-person personal pronouns: “I”, “me”, “we”, “our”, or “us”.

Do not retain the hedging expressions “perhaps”, “possibly”, “might”, “may”, “seems”, “appears
to”, “likely”, “presumably”, “arguably”, “I think”, “I believe”, or “could be”. Do not replace
these with synonymous conversational hedges. Preserve any substantive uncertainty,
conditionality, or probability judgment using direct, impersonal wording.

Do not use conversational backtracking expressions, including “wait”, “actually”, “on second
thought”, “let me reconsider”, “oh wait”, “scratch that”, or “I made a mistake”. Preserve the
underlying change in reasoning declaratively.

Do not add confidence, rhetorical emphasis, rhetorical questions, or commentary about the
rewriting task. Otherwise preserve the original structure and technical formatting as closely as
possible.

## Output

Output only the rewritten transcript, with no preamble or commentary.

Do not add, delete, or move any answer-labelled line. Copy every answer-labelled line exactly,
character for character, in its original position. Preserve any content that follows the final
answer-labelled line; do not move that line to the end of the transcript.
