# Output-conditioned token manifolds — HANDOFF 2026-08-17

## One-line state

**Critical correction, 2026-08-18:** the original 100-pair activation pilot is superseded for
science because its order-preserving limit contained only four Qwen items and five Llama items.
It remains an engineering/method-calibration artefact only. A tested deterministic item-round-robin
sampler is committed; corrected item-balanced storage estimates and recordings use the new immutable
tag `item_balanced_core100_v2`. No semantic feature or causal claim may use the old pilot.

## Why the first pilot is invalid for inference

`limit_pairs` previously took the first 100 pair IDs after fold selection. The pairs parquet is
item-clustered, so the nominally item-grouped discovery cells actually contained:

- Qwen fold A: 100 pairs / 200 prompts / **4 items**;
- Llama fold B: 100 pairs / 200 prompts / **5 items**.

This explains why nominal cross-validation halves contained only two items. All results below this
correction—including late ridge/CEBRA/BSF/SAE values—are retained as pipeline calibration, not
evidence. The fix is `item_round_robin_v1`: whole pair IDs are selected in deterministic rounds
across every available item while original row order is preserved. New manifests record
`limit_sampling=item_round_robin_v1`; estimates report `n_pairs` and `n_items` explicitly.

The calibration nevertheless exposed and fixed real implementation defects:

- fold-local residual and latent-code normalization;
- nested item-grouped regularization over a sufficiently wide ridge grid;
- CEBRA `delta_normal` conditioning and nonlinear KNN decoding;
- semantic-blind prompt-balanced token sampling;
- separate per-layer state and prompt-local change surfaces;
- exact-token and strictly item-cross-fitted trace exporters;
- genuine BatchTopK rather than the rejected inference-only per-sample TopK control;
- post-hoc capability/length/position/error labels kept out of representation fitting;
- held-out inherited-versus-new cross-layer decomposition.

Runpod pod `s4163rowfu5yc6` was created briefly, received no data, and was terminated. No new Runpod
volume exists and account spend returned to the pre-existing baseline. SWC remains the execution
environment.

## Corrected item-balanced core now recording

Compute estimates completed cleanly with the new sampler:

| Model/fold | estimate job | pairs | prompts | items | tokens | residual bytes |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B / A | 3392176 | 100 | 200 | 20 | 325,606 | 32,008,372,224 |
| Llama-3.2-3B-Instruct / B | 3392177 | 100 | 200 | 20 | 334,728 | 24,678,825,984 |

Corrected immutable recording jobs:

| Model/fold | job | destination |
|---|---:|---|
| Qwen3-8B / A | 3392178 | `token_manifold/item_balanced_core100_v2/record_Qwen3-8B_A` |
| Llama-3.2-3B-Instruct / B | 3392179 | `token_manifold/item_balanced_core100_v2/record_Llama-3.2-3B-Instruct_B` |

Check these jobs roughly every 30 minutes, per user preference. Do not delete or overwrite old or
new Ceph artefacts.

## Research questions that bind the corrected analysis

1. Which exact tokens change anonymous internal decision representations?
2. Which post-hoc external transcript properties those objects encode, especially independent
   capability/strength versus length, position/order and other biases?
3. In errors where the judge selects the longer but weaker transcript, when does capability
   information appear, when does the bias representation appear, and why does it take over?
4. Can layer-X dimensions predict output-associated layer-Y geometry? Decompose layer Y into an
   inherited/predictable component and a residual; define new information by incremental held-out
   gap prediction beyond layer X, not residual predictivity alone.
5. Which objects causally change the judge's own logit gap, and how do pairs interact non-additively?
6. Which objects replicate across Qwen/Llama, conditional on each model exhibiting the corresponding
   behaviour?

Capability, length, position, style and correctness labels are prohibited during representation
fitting. They enter only after anonymous objects are frozen. Semantic names require an independently
measurable rule, confirmation-fold re-extraction/alignment and causal controls.

## Scientific question

Discover residual-stream geometry that predicts each base judge's own continuous signed logit gap
without supplying semantic feature names. Then localise the discovered objects to exact transcript
tokens/phrases, name their external correlates post hoc, independently re-extract and align them, and
test causal effects and interactions. Qwen3-8B and Llama-3.2-3B-Instruct are reciprocal discovery and
replication models. Fine-tuned models come only after the base analysis is frozen.

Design: `docs/superpowers/specs/2026-08-17-output-conditioned-token-manifolds-design.md`

Plan: `docs/superpowers/plans/2026-08-17-token-manifold-bakeoff.md`

## Superseded pilot timeline results — engineering only

Both jobs used 200 prompts (100 pair IDs, both orders), but only four/five items. They used immutable float16 residual shards at 12
layers, item-grouped five-fold OOF ridge, equal prompt weights, fold-local mean/RMS scaling, and a
prompt-level shuffled-gap null.

| Model | job | peak token | frozen physical layer | held-out R2 | shuffled R2 | interpretation |
|---|---:|---|---:|---:|---:|---|
| Qwen3-8B base, fold A | 3391796 | answer readout | 29 | 0.4162 | -0.0170 | strong late readout locator |
| Llama-3.2-3B-Instruct base, fold B | 3391797 | answer readout | 25 | 0.0381 | -0.0411 | weak late readout locator |

Qwen readout R2 rises from -0.118 at layer 23 to 0.403 at 26, peaks at 0.416 at 29, stays 0.411 at
33, then falls to 0.036 at 36. Llama rises from -0.070 at layer 15 to -0.005 at 18, crosses zero at
23, and peaks at 0.038 at 25. All transcript-token cells remain below zero in both models. Therefore
the current evidence localises an output decision near the readout but does not yet explain which
transcript tokens determine it; that is the purpose of the nonlinear/sparse method gate now running.

Compact local summaries:

- `judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/timeline_qwen_v2.csv`
- `judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/timeline_llama_v2.csv`

The corresponding OOF tables remain on Ceph because they are about 331 MB and 306 MB.

## Superseded mixed-token bake-offs — engineering only

| Model | job | layer | new Ceph output directory |
|---|---:|---:|---|
| Qwen3-8B base | 3391805 | 29 | `bakeoff_qwen_v2_scientific_L29` |
| Llama-3.2-3B-Instruct base | 3391806 | 25 | `bakeoff_llama_v2_scientific_L25` |

Each job ran the pre-declared full grid: five seeds; CEBRA dimensions 1/2/4/8/16 at 10,000
iterations; BSF block sizes 1/2/4/8; and matched BatchTopK SAE. Outputs are `bakeoff.csv` and
`bakeoff.pt`. Do not interpret the earlier `*_quick` directories: they used smoke settings and layers
selected from the superseded timeline.

User preference: check these long jobs no more than once per hour.

## SWC paths and commands

```bash
C=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
ROOT=$C/staging/whitebox/token_manifold/base_manifold_pilot100_v1

ssh swc 'sacct -j 3391805,3391806 --format=JobID,State,Elapsed,ExitCode,NodeList -X'

# After each job is COMPLETED, pull only the compact CSV first.
rsync -a swc:$ROOT/bakeoff_qwen_v2_scientific_L29/bakeoff.csv \
  judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/bakeoff_qwen_v2_scientific_L29.csv
rsync -a swc:$ROOT/bakeoff_llama_v2_scientific_L25/bakeoff.csv \
  judge-whitebox/out/token_manifold/base_manifold_pilot100_v1/bakeoff_llama_v2_scientific_L25.csv
```

Never overwrite or delete Ceph files. Check local pull destinations are absent before `rsync`.

## Code and verification

Latest manifold commit: `d4fb7d7` (`token manifolds: normalize ridge folds`). The staged SWC
`token_manifold_discover.py` SHA-256 matches local:
`ff00a6bdd378da8edd53b7278f2509660b45f8db5fb092402065390cf5f0a677`.

- Focused manifold suite: 30 passed.
- Full whitebox suite: 199 passed, 4 optional skips, 2 existing constant-input warnings.

The isolated `.venv-manifold` intentionally lacks general project dependencies such as
`python-dotenv`; use it for manifold tests and `uv run pytest judge-whitebox/tests -q` for the full
project suite.

## Superseded and failed artefacts — do not delete

The first timeline CSV/OOF pairs are invalid for interpretation because ridge normalization was not
fitted within each CV fold:

- `$ROOT/timeline_qwen.csv`
- `$ROOT/timeline_qwen_oof.csv`
- `$ROOT/timeline_llama.csv`
- `$ROOT/timeline_llama_oof.csv`

Earlier failed/cancelled jobs left logs only:

- `$C/logs/wb-tm-est-llama-3391635.out`
- `$C/logs/wb-tm-envcheck-3391639.out`
- `$C/logs/wb-tm-overlaycheck-3391643.out`
- `$C/logs/wb-token-manifold-3391650.out`
- `$C/logs/wb-token-manifold-3391651.out`

List these for user cleanup if wanted; never remove them autonomously.

## Next gate

When both bake-offs finish, compare held-out gap decoding, reconstruction, active-code budget and
seed stability. Retain a complementary pair if CEBRA best captures decision geometry while BSF/SAE
provides the better inverse residual map. Do not assign semantic names yet. Only after this empirical
review should Plan 2 implement phrase localisation, post-hoc naming, independent re-extraction,
alignment, causal interventions and interactions.
