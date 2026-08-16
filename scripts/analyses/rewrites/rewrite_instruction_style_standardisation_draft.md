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
- “Products could result from route A or route B” can become “Products from route A or route B
  are among the possible outcomes”.
- “PFA may also capture transient interactions” can become “PFA has the capacity to capture
  transient interactions; whether it does so here is unresolved”.
- “The discrepancy may arise from X” can become “X is one possible source of the discrepancy”.
- “G2 may be a transcription factor” can become “A transcription factor is one candidate
  identity for G2; the identity is not established”.
- “The component is likely methylcyclopentane” can become “Methylcyclopentane has the strongest
  support among the candidate identities for the component”.
- “The peaks would likely not persist” can become “Among the possible outcomes, non-persistence
  of the peaks has the stronger support”. The word “would” alone does not preserve the original
  probability judgment.
- “A rough estimate might suggest X” can become “The rough estimate provides limited support for
  X”. It must not become “The rough estimate indicates X”.
- “The observation suggests that the reaction might not be straightforward” can become “The
  observation provides evidence against a straightforward reaction without establishing that
  conclusion”. It must not become “The observation suggests that the reaction is not
  straightforward”: embedding a categorical claim under “suggests” does not preserve the
  original modal force.

For every targeted hedge, first determine whether it changes the proposition's epistemic force.
If deleting it would turn a possibility, candidate, conditional result, or best-supported option
into a fact, the uncertainty is substantive and must be restated explicitly in direct, impersonal
language. Only remove a hedge without replacement when it is purely speaker-centred and its
removal leaves the proposition unchanged. When the distinction is unclear, preserve the
uncertainty.

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

Before outputting, compare the original and rewrite proposition by proposition. For every modal
or probability expression, verify that the rewritten proposition has the same epistemic strength.
Never replace “may”, “might”, “could”, “could be”, or “likely” with an unqualified categorical
verb such as “is”, “does”, “results”, “arises”, or “captures”. If the draft does so, revise it to
state the original possibility, conditionality, or relative support directly and impersonally.
Likewise, do not treat “would”, “suggests”, “indicates”, “can”, or “is expected to” as automatic
substitutes for a probability judgment: the rewrite must still state whether the original claim
was merely possible, weakly supported, or the best-supported alternative.

## Output

Output only the rewritten transcript, with no preamble or commentary.

Do not add, delete, or move any answer-labelled line. Copy every answer-labelled line exactly,
character for character, in its original position. Preserve any content that follows the final
answer-labelled line; do not move that line to the end of the transcript.
