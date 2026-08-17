# Output-conditioned token-manifold discovery — design

**Date:** 2026-08-17
**Status:** approved direction; written specification for review
**Models:** base Qwen3-8B and base Llama-3.2-3B-Instruct
**Primary target:** each judge's own continuous signed answer-logit gap, `l1 - l2`

## Objective

Discover, without pre-naming transcript properties, the residual-stream geometry that predicts a
base judge's verdict; trace that geometry back to the token phrases that create or modify it; name
external transcript properties only after discovery; and establish their causal effects and
interactions. Discoveries from one base model are confirmed in the other only when the corresponding
behaviour is present. Fine-tuned checkpoints are analysed later with the base-model procedure locked.

This inverts the earlier analysis direction. Instead of predicting verdicts from hand-authored
transcript features, it first predicts the verdict from internal states and then asks what external
properties the discovered internal geometry represents.

## Terminology and claim discipline

The discovery is **output-supervised but semantically unconstrained**:

- The supervisory target is the model's own logit gap, including biased and incorrect decisions.
- No capability, correctness, position, length, style, instruction or reasoning label shapes the
  discovery representation.
- Unsupervised reconstruction methods may be used, but output association selects the components of
  interest.
- CEBRA-Behaviour/Hybrid is target-guided manifold estimation, not label-free learning.

Claims remain ordered:

1. **Decodability:** a latent representation predicts held-out logit gaps.
2. **Association:** a discovered coordinate/block covaries with the output.
3. **Causal sensitivity:** an internal intervention moves the output selectively.
4. **Necessity/sufficiency:** removal or isolated restoration destroys or recreates the effect.
5. **Interaction:** a joint intervention differs from the sum of single interventions.
6. **Mediation/path:** deferred unless sender-to-receiver patching is performed.

No semantic feature is reported solely from examples, correlation, an embedding plot or an automated
label. No manifold is claimed solely from low-dimensional visualization.

## Scope

### Included

- Base Qwen3-8B and base Llama-3.2-3B-Instruct on the same GPQA pair prompts, both orders.
- Exact tokenizer positions for recording and screening.
- Deterministically merged contiguous positions for human-readable phrase interventions.
- Residual-stream trajectories at selected layers, model logit gaps, token offsets and prompt metadata.
- Linear decoder baseline, CEBRA-Behaviour/Hybrid, block-sparse featurizers, and a matched SAE plus
  inverse-Ising manifold-recovery baseline.
- Internal feature/manifold interventions, phrase interventions and pairwise interaction tests.
- Independent naming, re-extraction and cross-model confirmation.
- A later, locked base-versus-FT comparison.

### Excluded from discovery

- Fine-tuned checkpoints.
- Hand-authored semantic axes.
- Paid APIs or automated external-feature naming.
- Higher-order interactions beyond pairs.
- Edge/path claims.
- Qwen3-0.6B scientific evidence; synthetic models provide engineering tests.

## Data partitioning and leakage control

All rows for one GPQA item—including every model pair and both orders—remain in one group.

1. Deterministically divide the 42 items into balanced outer folds A and B, balancing row counts and
   answer labels with a recorded seed.
2. **Direction 1:** discover and name on Qwen fold A; confirm within Qwen on fold B and across models
   on Llama fold B.
3. **Direction 2:** discover and name on Llama fold B; confirm within Llama on fold A and across models
   on Qwen fold A.
4. Within each discovery fold, use item-grouped cross-validation for layer selection, method settings,
   output association and causal-candidate ranking.
5. The confirmation cells for each direction remain sealed until the feature name, external operational
   definition, layer, token/phrase rule, intervention dose and sign have been frozen. Analyses in the
   two directions keep separate hypothesis ledgers so a feature named in one direction cannot be
   retrofitted to the other discovery cell.
6. A model-specific null is not treated as failed replication if the named external property has no
   behavioural association with that model's logit gap on the confirmation fold. That behavioural
   gate is evaluated without changing the intervention specification.

The two directional analyses are reported separately before any pooled summary.

## Recording architecture

The current `record_resid.py` stores only the last-token state. A new token recorder will retain the
causal residual state at every tokenizer position for a bounded core set and stream compact codes over
the full evaluation set.

### Core activation tier

- Record all token positions for both orders on a balanced subset of discovery-fold pairs.
- Start from 12 evenly spaced layers per model. Existing last-token decoder curves may add adjacent
  layers around a transition, but may not remove the early, middle or late controls.
- Store float16 residual shards plus model ID, layer, pair UID, item, order, token ID, character offset,
  structural span, phrase boundary and output logits.
- Shards are immutable and content-addressed. A manifest records shapes, token counts, bytes, model
  revision, prompt hash and dataset hash.
- Before any SWC launch, a local dry run computes the exact storage estimate from observed token counts;
  the user reviews it against the ceph quota.

### Full-evaluation scalar tier

After featurizers are fitted, stream the full evaluation set and save only latent codes, block norms,
selected coordinates, decoder predictions, attributions and metadata. Full residual tensors are not
retained for every row.

Longer prompts receive no extra statistical weight merely because they contain more tokens. Training
samplers weight prompts equally, then sample positions within prompt. Temporal samplers never cross a
prompt boundary.

## Stage 1: decision-information timeline

Fit item-cross-validated ridge and small bottleneck decoders from residual states to the final signed
logit gap. These are locators and baselines, not semantic explanations.

Outputs:

- Held-out `R²` by layer and token position/phrase.
- The token-time point where verdict information first becomes decodable.
- Linear-versus-nonlinear decoding gain.
- Base-order trajectories showing whether a decision is present before, during or after each transcript.

Layers carried into manifold discovery must include the earliest reproducible rise, the peak, and early
and late controls. Layer selection is frozen before method comparison.

## Stage 2: manifold and feature discovery

All discovery methods consume the same normalized residual shards and the same splits. Per-layer mean
and RMS are fitted on the discovery training data only.

### CEBRA

Fit two target-guided variants using continuous logit gap:

- **CEBRA-Behaviour:** similarity is defined by the judge's output gap.
- **CEBRA-Hybrid:** combines output-gap similarity with within-prompt token-time continuity.

Evaluate output dimensions `{2, 4, 8, 16}` and at least five random seeds. The representation is judged
by held-out gap decoding, Procrustes consistency across seeds, neighbourhood overlap, topology and
token-trajectory stability. A dimension is retained only when it adds reproducible structure beyond a
monotonic one-dimensional encoding of the gap.

CEBRA has no guaranteed inverse map to residual space. A CEBRA manifold is not eligible for causal
intervention until either:

- a held-out local decoder maps intrinsic coordinates back to residual states with acceptable error and
  density, or
- the manifold is matched to a reconstructive BSF/SAE component that supplies the residual-space map.

### Block-Sparse Featurizer

Fit the released Goodfire block-sparse family with matched normalization and reconstruction budgets:

- block dimensions `{1, 2, 4, 8}`;
- block-TopK sparsity selected by held-out minimum description length at fixed distortion;
- Grassmannian and vanilla variants in the bake-off; group lasso only if neither is stable.

Block norm means feature presence; within-block coordinates describe position on the candidate
manifold. Stable rank estimates whether the block is genuinely multidimensional. Block dimension 1 is
the direction-based baseline.

### SAE plus inverse-Ising grouping

Fit a BatchTopK SAE matched to the BSF's reconstruction distortion and average active-feature count.
Recover candidate manifolds by fitting pairwise inverse-Ising couplings on binarized feature codes and
running both Leiden and spectral community detection. Raw co-activation and decoder cosine are retained
as negative baselines because they confound marginal firing and need not recover manifold structure.

### Method-selection gate

No method wins on visualization. Compare:

- held-out reconstruction distortion and description length;
- held-out logit-gap decoding from latent codes;
- stability across seeds and item splits;
- neighbourhood/topology preservation;
- localisation to reproducible token phrases;
- availability and fidelity of an inverse residual map;
- selective causal effect versus matched controls.

CEBRA may best describe decision geometry while BSF best supplies causal coordinates. The design permits
that complementary result rather than forcing one global winner.

## Stage 3: output association and token localisation

On discovery data only:

1. Rank CEBRA coordinates/regions, BSF blocks and SAE-manifold clusters by item-cross-validated
   association with signed logit gap.
2. For reconstructive codes inserted into the model, compute contrastive-logit attribution
   `activation × d(l1-l2)/d(code)` at every token position.
3. Require agreement between held-out association, token attribution and at least one causal smoke
   intervention before naming.
4. Merge adjacent tokenizer positions into phrases using character offsets and a frozen contiguity rule.
5. Retain positive and negative examples, coordinate traversals and order-swapped trajectories.

The initial candidate budget is bounded: at most eight latent objects and ten phrase loci per object
advance to pairwise causal testing. Ranking and tie-breaking rules are frozen on discovery data.

## Stage 4: reverse interpretation and naming

Naming is hypothesis generation, followed by a separate extraction test.

1. Show top and bottom phrase examples and manifold traversals from the discovery fold with output gap,
   item identity and model preference hidden.
2. Give a candidate a descriptive name only if its examples support a concrete external property that
   can be measured without model activations or verdicts.
3. Freeze a written annotation/measurement rule, including counterexamples and abstention criteria.
4. Apply the rule to the sealed confirmation texts. If the property cannot be operationalized, retain an
   anonymous manifold ID and make no semantic claim.
5. Re-extract the named property from confirmation activations without using logit gaps.

Alignment to the original output-guided object is evaluated within each model using the appropriate
geometry:

- cosine only for a one-dimensional direction;
- principal angles for a block/subspace;
- Procrustes `R²`, neighbourhood overlap and geodesic-distance correlation for a curved manifold;
- token/phrase support overlap for localisation.

## Stage 5: causal effects and interactions

### Internal interventions

- **BSF/SAE delta intervention:** replace selected code values and reconstruct
  `h' = decode(z') + (h - decode(z))`, preserving the original reconstruction error.
- **Manifold intervention:** move in intrinsic coordinates and map back through a held-out local
  parameterization; reject points outside the training-density support.
- **Natural activation patch:** where inverse fidelity is insufficient, patch a matched natural
  activation from a nearby manifold coordinate rather than extrapolating.

Every intervention is compared against matched-norm random tangent/subspace interventions,
shuffled-token controls and a linear off-manifold path. Readouts are signed logit gap, pick flips,
answer-token probability mass, next-token KL and control-prompt NLL.

### Phrase and latent interactions

For intact output `y0`, single interventions `yA`, `yB`, and joint intervention `yAB`, define:

`I(A,B) = yAB - yA - yB + y0`.

- `I ≈ 0`: additive at the tested dose.
- Same-direction `I`: synergy.
- Opposing `I`: redundancy, competition or self-repair; distinguish these using coordinate trajectories
  and restoration tests.

Test both phrase-pair and latent-object-pair interactions, but only among discovery-ranked candidates.
Higher-order combinations are excluded. Report item-clustered bootstrap intervals and control the false
discovery rate across the frozen candidate family.

Necessity requires a dose curve and a restoration test; a single zero-ablation is insufficient.

## Stage 6: cross-model confirmation

Qwen and Llama residual vectors are never compared directly: their dimensions, tokenizers and bases
differ.

For a named Qwen discovery:

1. Apply the frozen external property rule to Llama confirmation texts.
2. Test whether that property is behaviourally associated with Llama's logit gap.
3. Independently extract its Llama representation.
4. Test whether it lies in a Llama output-associated manifold/block and reproduces the frozen causal
   intervention and interaction signs.

Reverse the procedure for Llama discoveries. Cross-model success is shared external meaning, internal
alignment within each model and matched causal behaviour—not decoder-vector cosine across models.

After independent confirmation, optional multi-session CEBRA may place Qwen and Llama into a shared
output-conditioned coordinate system. Because the common logit index imposes alignment, this is a
descriptive synthesis, not independent replication evidence.

## Stage 7: later fine-tuning comparison

Only after base discoveries are locked:

1. Apply frozen base featurizers/encoders to FT activations.
2. Measure changes in prevalence, coordinate distribution, token localisation, output association,
   causal effect and pairwise interaction.
3. Fit the same discovery families anew on FT activations and align them to base objects.
4. Classify each base object as preserved, strengthened, suppressed, rotated/warped, relocated or
   replaced.

No new semantic names from FT enter the primary base-versus-FT comparison.

## Statistical and causal controls

- Group all uncertainty and cross-validation by item, never row.
- Keep both slot orders together and report order-specific trajectories.
- Use multiple featurizer seeds and report feature/manifold stability, not a cherry-picked run.
- Compare against label-shuffled logit gaps, random subspaces, random phrase spans and off-manifold paths.
- Stratify pick flips by baseline margin.
- Require intervention selectivity: a target gap change without broad KL/NLL corruption.
- Treat output-conditioned decoding as evidence of information, not use; causal intervention is required.
- Treat a cross-model null as expected when the external behavioural gate is absent.

## Engineering boundaries

The implementation remains inside `judge-whitebox/` and reuses `wbprompts`, `wbpairs`, `verdicts`,
`wbconfig` and the existing hook conventions.

Planned responsibilities:

- `token_manifold_record.py`: sharded per-token activation recording and manifests.
- `token_manifold_discover.py`: decoder screen, CEBRA, BSF and SAE/Ising adapters, method gates.
- `token_manifold_intervene.py`: reconstructive/manifold patches, controls and interaction runner.
- `token_manifold_report.py`: alignment, naming tables, bootstrap summaries and export.
- `sbatch/token_manifold.sbatch`: recorder/fitter/intervention stages with skip-if-done outputs.
- Matching small test modules under `judge-whitebox/tests/`.

External research code is pinned, not copied loosely:

- CEBRA `v0.6.0`, Apache-2.0.
- Goodfire `block-sparse-featurizer`, recorded commit, MIT.
- Goodfire `sae-manifold`, recorded commit, MIT.

No package install or heavy process runs on the SWC gateway. Environment construction is user-run or a
compute job. Ceph data files are never overwritten or deleted.

### Implementation decomposition

This specification is too broad for one uninterrupted implementation plan. It is delivered through
three sequential, independently testable plans:

1. **Recorder and method bake-off:** splits, manifests, per-token shards, decoder timeline, CEBRA/BSF/
   SAE adapters, synthetic recovery tests, storage estimate and a small discovery-fold comparison.
   This is the plan implemented first.
2. **Locked discovery and causal confirmation:** full discovery-fold fitting, token/phrase localisation,
   naming ledger, re-extraction, internal interventions and pairwise interaction tests, followed by the
   sealed within- and cross-model confirmations.
3. **Fine-tuning comparison:** frozen base encoders plus matched FT rediscovery and geometry/effect
   alignment.

Each later plan begins only after the previous plan's empirical gate is reviewed. This avoids building
intervention machinery around a discovery representation that fails stability or inversion tests.

## Testing and verification

CPU-only synthetic tests must cover:

- item/group split isolation and prompt-boundary-safe temporal sampling;
- activation shard round-trip and manifest/hash validation;
- a linear latent where dimension 1 wins;
- a curved manifold where CEBRA/BSF beats the linear baseline;
- a sparse mixture where BSF/SAE recovery is known;
- inverse-map fidelity and exact no-op reconstruction;
- intervention restoration and matched-norm controls;
- zero interaction for an additive toy model and known non-zero interaction for a nonlinear toy model;
- tokenizer-to-phrase merging across split subwords;
- cross-model alignment logic with different residual dimensions.

Real-model gates proceed in order:

1. two-row smoke with exact logits and no-op parity;
2. tiny base-model recording with verified storage estimate;
3. discovery-fold method bake-off;
4. frozen confirmation;
5. only then full scalar streaming and causal interaction runs.

## Deliverables

- Layer × token-time decision-decoding maps.
- Output-associated latent manifold/block catalogue with anonymous IDs before naming.
- Phrase viewers showing where each object activates and how it evolves through the prompt.
- Naming and independent re-extraction alignment tables.
- Single and pairwise causal-effect tables with interaction classifications and controls.
- Symmetric Qwen→Llama and Llama→Qwen confirmation report.
- A later base→FT change report using the locked base objects.

## References informing method choice

- Schneider, Lee & Mathis, *Learnable latent embeddings for joint behavioural and neural analysis*,
  Nature 2023: https://www.nature.com/articles/s41586-023-06031-6
- Bhalla et al., *Do Sparse Autoencoders Capture Concept Manifolds?*, 2026:
  https://arxiv.org/abs/2604.28119
- Fel et al., *Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds*, 2026:
  https://arxiv.org/abs/2606.25234
- Wurgaft et al., *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and
  Behavior*, 2026: https://arxiv.org/abs/2605.05115
- Goodfire released implementations:
  https://github.com/goodfire-ai/block-sparse-featurizer and
  https://github.com/goodfire-ai/sae-manifold
