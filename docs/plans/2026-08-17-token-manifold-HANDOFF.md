# Output-conditioned token manifolds — HANDOFF 2026-08-17

## One-line state

The base-model per-token residual recordings and corrected item-grouped decoder timelines are
complete. Qwen has a strong late readout-position signal; Llama has a weak late one; neither model
yet has positive linear decoding at transcript tokens. Full five-seed method bake-offs are submitted
on SWC at the frozen peak layers.

## Scientific question

Discover residual-stream geometry that predicts each base judge's own continuous signed logit gap
without supplying semantic feature names. Then localise the discovered objects to exact transcript
tokens/phrases, name their external correlates post hoc, independently re-extract and align them, and
test causal effects and interactions. Qwen3-8B and Llama-3.2-3B-Instruct are reciprocal discovery and
replication models. Fine-tuned models come only after the base analysis is frozen.

Design: `docs/superpowers/specs/2026-08-17-output-conditioned-token-manifolds-design.md`

Plan: `docs/superpowers/plans/2026-08-17-token-manifold-bakeoff.md`

## Corrected timeline results

Both jobs used 200 prompts (100 pair IDs, both orders), immutable float16 residual shards at 12
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

## Scientific bake-offs now submitted

| Model | job | layer | new Ceph output directory |
|---|---:|---:|---|
| Qwen3-8B base | 3391805 | 29 | `bakeoff_qwen_v2_scientific_L29` |
| Llama-3.2-3B-Instruct base | 3391806 | 25 | `bakeoff_llama_v2_scientific_L25` |

Each job runs the pre-declared full grid: five seeds; CEBRA dimensions 1/2/4/8/16 at 10,000
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
